import getpass
import os
import socket
import sys
import time

from . import __version__


def get_cf_global_attrs(**attrs):

    if 'history' not in attrs:
        attrs['history'] = 'Created: {}'.format(time.ctime(time.time()))

    if 'insitution' not in attrs:
        attrs['institution'] = 'CarbonPlan'

    if 'source' not in attrs:
        attrs['source'] = sys.argv[0]

    if 'hostname' not in attrs:
        attrs['hostname'] = socket.gethostname()

    if 'username' not in attrs:
        attrs['username'] = os.getenv('JUPYTERHUB_USER', getpass.getuser())

    if 'version' not in attrs:
        attrs['version'] = __version__

    return attrs
