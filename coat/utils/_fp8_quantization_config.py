from dataclasses import dataclass

from transformers import PretrainedConfig


@dataclass
class QuantizationConfig:
    # Quantize Activation
    quantize_model: str = "true"
    symm: bool = True
    epsilon: float = 1e-10
    fabit: str = "E4M3"
    fwbit: str = "E4M3"
    fobit: str = "E4M3"
    babit: str = "E5M2"
    bwbit: str = "E5M2"
    bobit: str = "E5M2"
    group_size: int = 16
    pad_to_multiple_of: int = 8
    weight_memory_efficient: bool = True

    # Quantize Optimizer States
    first_order_expansion: str = "false"
    second_order_expansion: str = "false"
    first_order_bit: str = "BF16"
    second_order_bit: str = "BF16"
    qgroup_size: int = 128
    expand_min: int = 16

    # Legacy
    row_blocksize: int = -1
    col_blocksize: int = -1

    def __init__(
        self,
        quantize_model: str = "true",
        symm: bool = True,
        epsilon: float = 1e-10,
        fabit: str = "E4M3",
        fwbit: str = "E4M3",
        fobit: str = "E4M3",
        babit: str = "E5M2",
        bwbit: str = "E5M2",
        bobit: str = "E5M2",
        group_size: int = 16,
        pad_to_multiple_of: int = 0,
        weight_memory_efficient: bool = True,
        first_order_expansion: str = "false",
        second_order_expansion: str = "false",
        first_order_bit: str = "BF16",
        second_order_bit: str = "BF16",
        qgroup_size: int = 128,
        expand_min: int = 16,
        row_blocksize: int = -1,
        col_blocksize: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.quantize_model = quantize_model
        self.symm = symm
        self.epsilon = epsilon
        self.fabit = fabit
        self.fwbit = fwbit
        self.fobit = fobit
        self.babit = babit
        self.bwbit = bwbit
        self.bobit = bobit
        self.group_size = group_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.weight_memory_efficient = weight_memory_efficient

        self.first_order_expansion = first_order_expansion
        self.second_order_expansion = second_order_expansion
        self.first_order_bit = first_order_bit
        self.second_order_bit = second_order_bit
        self.qgroup_size = qgroup_size
        self.expand_min = expand_min
        
        self.row_blocksize = row_blocksize
        self.col_blocksize = col_blocksize