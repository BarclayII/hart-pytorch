
class DynamicConvFilter(nn.Module):
    '''
    generator: Filter generator module to generate filter weights
        takes input argument @h and produces a Tensor with size
            (batch_size, out_channels, in_channels, kernel_size)
        if bias is True, produces an additional Tensor with size
            (batch_size, out_channels)
    the rest are the same as Conv1D, Conv2D, Conv3D
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 transposed=False,
                 output_padding=0,
                 ):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.transposed = transposed
        self.output_padding = output_padding

        def _conv(x, w, bias=None, stride=1, padding=0, dilation=1,
                  groups=1, output_padding=0, conv=None,
                  conv_transpose=None, transposed=False):
            if transposed:
                return conv_transpose(x, w, bias, stride, padding,
                                      output_padding, groups, dilation)
            else:
                return conv(x, w, bias, stride, padding, dilation, groups)

        if len(self.kernel_size) == 1:
            conv_transpose = F.conv_transpose1d
            conv = F.conv1d
        elif len(self.kernel_size) == 2:
            conv_transpose = F.conv_transpose2d
            conv = F.conv2d
        elif len(self.kernel_size) == 3:
            conv_transpose = F.conv_transpose3d
            conv = F.conv3d
        else:
            raise ValueError('only 1D, 2D and 3D filters are supported')

        self.f = partial(_conv, conv=conv, conv_transpose=conv_transpose,
                         transposed=transposed)

    def forward(self, x, h, g):
        '''
        x: (batch_size, nchannels, ...)
        h: anything passed into g()
        g: a callable
        '''
        if self.bias:
            w, b = g(h)
        else:
            w = g(h)
            b = None

        x_shape = list(x.size())
        batch_size, nchannels = x_shape[0:2]
        x_trailing_shape = x_shape[2:]

        # push batch dimension to channel dimension
        x = x.view(1, batch_size * nchannels, *x_trailing_shape)
        w = w.view(batch_size * self.out_channels, self.in_channels,
                   *self.kernel_size)
        if b is not None:
            b = b.view(batch_size * self.out_channels)

        # do a grouped convolution as batch-wise convolution with individual
        # filters
        y = self.f(x, w, b, self.stride, self.padding, self.dilation,
                   batch_size, self.output_padding)
        y_trailing_shape = y.size()[2:]
        y = y.view(batch_size, self.out_channels, *y_trailing_shape)
        return y


class DynamicConvFilterGenerator(nn.Module):
    def __init__(self,
                 input_size,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=2,
                 nonlinearity=F.elu,
                 bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        output_size = in_channels * out_channels * np.prod(kernel_size)

        self.affines_w = nn.ModuleList()
        for i in range(num_layers):
            self.affines_w.append(
                    nn.Linear(
                        input_size if i == 0 else output_size, output_size
                        )
                    )
            input_size = output_size
        if bias:
            self.affines_b = nn.ModuleList()
            for i in range(num_layers):
                self.affines_b.append(
                        nn.Linear(
                            input_size if i == 0 else output_size, output_size
                            )
                        )

    def forward(self, x):
        batch_size = x.size()[0]
        w = x
        for i, m in enumerate(self.affines_w):
            w = m(w)
            if i != len(self.affines_w) - 1:
                w = self.nonlinearity(w)
        w = w.view(batch_size, self.out_channels, self.in_channels,
                   *self.kernel_size)

        if self.bias:
            b = x
            for i, m in enumerate(self.affines_b):
                b = m(b)
                if i != len(self.affines_b) - 1:
                    b = self.nonlinearity(b)
            b = b.view(batch_size, self.out_channels)

            return w, b
        else:
            return w
