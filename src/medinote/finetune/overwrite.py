from bitsandbytes.nn.modules import Linear8bitLt, Linear4bit
import torch.nn.init as init  # Importing the init module from PyTorch
from contextlib import contextmanager

def peft_initialization():

    def noop (x=None, *args, **kwargs):
        "Do nothing"
        return x

    @contextmanager
    def no_kaiming():
        old_iku = init.kaiming_uniform_
        init.kaiming_uniform_ = noop
        try: yield
        finally: init.kaiming_uniform_ = old_iku

    _old_8init = Linear8bitLt.__init__
    _old_4init = Linear4bit.__init__

    def _new_4init(self, input_features, output_features, bias=True,
                device=None, **kwargs):
        with no_kaiming():
            return _old_4init(self, input_features, output_features, bias=bias,
                            device=device, **kwargs)



    def _new_8init(self, input_features, output_features, bias=True, has_fp16_weights=True,
                memory_efficient_backward=False, threshold=0.0, index=None, device=None):
        with no_kaiming():
            return _old_8init(self, input_features, output_features, bias=bias, has_fp16_weights=has_fp16_weights,
                            memory_efficient_backward=memory_efficient_backward, threshold=threshold, index=index, device=device)

    Linear8bitLt.__init__ = _new_8init
    Linear4bit.__init__ = _new_4init
