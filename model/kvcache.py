from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)


_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.is_initialized and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """

    is_sliding = True

    def __init__(self, sliding_window: int):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]

        # Compute the full states
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        # Return the full states
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        if only_non_sliding:
            # Try to find next non-sliding, starting at `layer_idx`
            try:
                layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
            # In this case, we need to circle back to the beginning
            except ValueError:
                layer_idx = self.is_sliding.index(False)
        else:
            layer_idx = layer_idx if layer_idx < len(self.layers) else 0

        # Prefetch
        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            self.layers[layer_idx].prefetch()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if not (only_non_sliding and self.is_sliding[layer_idx]):
            self.layers[layer_idx].offload()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PretrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    ```
    """

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Optional[PretrainedConfig] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, (key_states, value_states) in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a DynamicLayer for each entry of the ddp_data
                if config is None:
                    layers.append(DynamicLayer())
                # Update the layer with the data
                _, _ = layers[layer_idx].update(key_states, value_states)

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]]) -> "DynamicCache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is None:
            logger.warning_once("past_key_values should not be None in from_legacy_cache()")
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    
    
    def keep(self, kv_len, block_size) -> "DynamicCache":
        """
        keep only the last `kv_len` before last 'block_size' tokens in the cache, and remove the rest.
        """
        total_len = self.get_seq_length()
        
        # Calculate the range to keep: [start_pos, end_pos)
        end_pos = total_len - block_size
        start_pos = max(0, end_pos - kv_len)
        
        # For each layer, slice the cache to keep only the specified range
        for layer in self.layers:
            if layer.is_initialized and layer.keys.numel() > 0:
                layer.keys = layer.keys[..., start_pos:end_pos, :]
                layer.values = layer.values[..., start_pos:end_pos, :]

        return self

  