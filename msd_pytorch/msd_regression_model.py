from msd_pytorch.msd_model import MSDModel
import torch.nn as nn


loss_functions = {"L1": nn.L1Loss(), "L2": nn.MSELoss()}


class MSDRegressionModel(MSDModel):
    """An MSD network for regression.

    This class provides helper methods for using the MSD network
    module for regression.

    Refer to the documentation of
    :class:`~msd_pytorch.msd_model.MSDModel` for more information on
    the helper methods and attributes.

    """

    def __init__(
        self,
        c_in,
        c_out,
        depth,
        width,
        *,
        dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        loss="L2",
    ):
        """Create a new MSD network for regression.

        :param c_in: The number of input channels.
        :param c_out: The number of output channels.
        :param depth: The depth of the MSD network.
        :param width: The width of the MSD network.
        :param dilations: `list(int)`

        A list of dilations to use. Default is ``[1, 2, ..., 10]``.  A
        good alternative is ``[1, 2, 4, 8]``. The dilations are
        repeated when there are more layers than supplied dilations.

        :param loss: `string`

        A string describing the loss function that should be
        used. Currently, the following losses are supported:

        * "L1" - ``nn.L1Loss()``
        * "L2" - ``nn.MSELoss()``

        :returns:
        :rtype:

        """
        super().__init__(c_in, c_out, depth, width, dilations)

        self.criterion = loss_functions.get(loss)
        if self.criterion is None:
            raise ValueError(
                "The loss must be one of {list(loss_functions.keys())}. Supplied {loss_function}. "
            )

        # Define the whole network:
        self.net = nn.Sequential(self.scale_in, self.msd, self.scale_out)
        self.net.cuda()

        # Train only MSD parameters:
        self.init_optimizer(self.msd)
