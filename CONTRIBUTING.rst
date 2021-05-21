.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ahendriksen/msd_pytorch/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.


Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ahendriksen/msd_pytorch/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Building from source
--------------------

Pytorch versions >= 1.7
~~~~~~~~~~~~~~~~~~~~~~

Use the following:

    $ CC=/opt/sw/gcc-7.3.0/bin/gcc CXX=/opt/sw/gcc-7.3.0/bin/g++ python setup.py emit_ninja



Pytorch versions < 1.7
~~~~~~~~~~~~~~~~~~~~~~

Use the following:

    $ CC=/opt/sw/gcc-7.3.0/bin/gcc GXX=/opt/sw/gcc-7.3.0/bin/ CXX=/opt/sw/gcc-7.3.0/bin/g++ python setup.py emit_ninja
