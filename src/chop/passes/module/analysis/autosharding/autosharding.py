

def alpa_intra_op_sharding_pass(model):

    """
    A lightweight implementation of the core algorithm from the Alpa paper: https://arxiv.org/abs/2201.12023
    """

    for name, cls in model.named_children():
        print(cls)

    return model, {}

def autosharding_analysis_pass(model):

    model = alpa_intra_op_sharding_pass(model)

    return model, {}