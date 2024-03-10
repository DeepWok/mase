import ...

def onnx_runtime_transform_pass(graph, pass_args="None"):
    onnx_runtime_session = ONNXRuntime()

    do_test = pass_args['do_test']
    if do_test == 'before' or do_test == 'both':
        ... #test pl performances

    if do_test == 'after' or do_test == 'both':
        ... #test onnx performances

    elif do_test == 'NA':
        pass

    else:
        raise Exception(f"Test argument not recognized; expected one in ['before','after','both','NA'], but {do_test} was received.")
    
    return graph, {}


class ONNXRuntime:
    def __init__():
        ...

    def pytorch_to_onnx(self, ):
        ...

    def load_onnx(self, ):
        '''Load .onnx model'''
        ...

    def test_performances(self, how="NA"):
        ...

        