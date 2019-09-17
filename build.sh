#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
rm -rf build/ dist/ sparseconvnet.egg-info
python setup.py install && python examples/hello-world.py
