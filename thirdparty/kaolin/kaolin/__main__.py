# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m kaolin方式直接执行。

Authors: wujinbo01(wujinbo01@baidu.com)
Date:    2023/11/03 15:08:42
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from kaolin.cmdline import main
sys.exit(main())
