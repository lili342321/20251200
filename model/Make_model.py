from vector_fields import *
from GCDE import *

def make_model(args):
    if args.model_type == 'type1':
        model = FFCM(args,  input_channels=args.input_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver,init_len=args.init_len)
        return model