rem Copyright 2016-present, Facebook, Inc.
rem All rights reserved.
rem
rem This source code is licensed under the BSD-style license found in the
rem LICENSE file in the root directory of this source tree.

rem export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
rem del works a bit differently from rm -rf, so I guess just delete these manually, as well as the pyd file
rem del "build/" "dist/" "sparseconvnet.egg-info"
python setup.py develop && python examples/hello-world.py
