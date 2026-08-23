from visdom import Visdom
import json 

class Visualizer(object):
    """ Visualizer
    """
    def __init__(self, port='13579', env='main', id=None):
        self.vis = Visdom(port=port, env=env)
        self.id = id
        self.env = env

    def vis_scalar(self, name, x, y, opts=None):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if self.id is not None:
            name = "[%s]"%self.id + name
        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)

                                           
        self.vis.line( X=x, Y=y, win=name, opts=default_opts, update='append')

    def vis_image(self, name, img, env=None, opts=None):
        """ vis image in visdom
        """
        if env is None:
            env = self.env 
        if self.id is not None:
            name = "[%s]"%self.id + name
        default_opts = { 'title': name }
        if opts is not None:
                default_opts.update(opts)
        self.vis.image( img=img, win=name, opts=opts, env=env )
    
    def vis_table(self, name, tbl, opts=None):

        tbl_str = "<table width=\"100%\"> "
        tbl_str+="<tr> \
                 <th>Term</th> \
                 <th>Value</th> \
                 </tr>"
        for k, v in tbl.items():
            tbl_str+=  "<tr> \
                       <td>%s</td> \
                       <td>%s</td> \
                       </tr>"%(k, v)

        tbl_str+="</table>"

        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)
        self.vis.text(tbl_str, win=name, opts=default_opts)

if __name__=='__main__':
    import numpy as np
    vis = Visualizer(port=35588, env='main')
    tbl = {"lr": 214, "momentum": 0.9}
    vis.vis_table("test_table", tbl)
    tbl = {"lr": 244444, "momentum": 0.9, "haha": "hoho"}
    vis.vis_table("test_table", tbl)

    vis.vis_scalar(name='loss', x=0, y=1)
    vis.vis_scalar(name='loss', x=2, y=4)
    vis.vis_scalar(name='loss', x=4, y=6)