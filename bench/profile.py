"""Profile the current MSDModule implementation and store the results
   in a database through sacred (if available).
"""

from msd_pytorch.msd_reg_model import (MSDRegressionModel, msd_ingredient)
from os import environ
from sacred import Experiment
from sacred.observers import MongoObserver
from timeit import default_timer as timer
import torch as t

ex = Experiment('Profile MSD', ingredients=[msd_ingredient])

mongo_enabled = environ.get('MONGO_SACRED_ENABLED')
mongo_user = environ.get('MONGO_SACRED_USER')
mongo_pass = environ.get('MONGO_SACRED_PASS')
mongo_host = environ.get('MONGO_SACRED_HOST')

if mongo_enabled == 'true':
    assert mongo_user, 'Setting $MONGO_USER is required'
    assert mongo_pass, 'Setting $MONGO_PASS is required'
    assert mongo_host, 'Setting $MONGO_HOST is required'

    mongo_url = 'mongodb://{0}:{1}@{2}:27017/sacred?authMechanism=SCRAM-SHA-1'.format(
        mongo_user, mongo_pass, mongo_host)

    ex.observers.append(MongoObserver.create(url=mongo_url, db_name='sacred'))


@ex.config
def cfg():
    img_sz = 128
    batch_sz = 1
    c_in = 1
    c_out = 1
    iterations = 2
    do_backward = True
    conv3d = False


@ex.automain
def profile(img_sz, batch_sz, c_in, c_out, iterations,
            do_backward, conv3d):
    size = (img_sz,) * (3 if conv3d else 2)

    model = MSDRegressionModel(c_in, c_out, conv3d=conv3d)
    input = t.randn(batch_sz, c_in, *size).cuda()
    target = t.randn(batch_sz, c_out, *size).cuda()

    t.cuda.synchronize()

    start = timer()

    for _ in range(iterations):
        model.set_input(input)
        model.set_target(target)
        if do_backward:
            model.learn()
        else:
            model.forward()

    t.cuda.synchronize()

    end = timer()

    print("Total time:         {}".format(end - start))
    print("Time per iteration: {}".format((end - start) / iterations))

    ex.log_scalar("Total time", end - start)
    ex.log_scalar("Time per iteration", (end - start) / iterations)
