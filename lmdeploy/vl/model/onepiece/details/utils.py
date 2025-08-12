# Copyright (c) Shopee. All rights reserved.
def check_and_raise(error_cls, flag, msg):
    if flag:
        return
    raise (error_cls(msg))
