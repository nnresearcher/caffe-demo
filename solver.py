
from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path):

    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.type = 'SGD'
    s.test_interval = 1000
    s.test_iter.append(250)
    s.max_iter = 10000
    s.base_lr = 0.01
    s.lr_policy = 'step'
    s.gamma = 0.9
    s.stepsize = 500
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.display = 500
    s.snapshot = 10000
    s.snapshot_prefix = 'tess'
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    return s