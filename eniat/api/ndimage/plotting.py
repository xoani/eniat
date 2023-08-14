from ...helper import *
import matplotlib.pyplot as plt


def get_outline(data, thr=0.999):
    from skimage import filters
    data = np.nan_to_num(data.copy())
    outline = filters.prewitt(data)
    
    z = (outline - outline.mean()) / outline.std()
    z[z < norm.ppf(thr)] = 0
    z = z.astype(bool).astype(float)
    z[z == 0] = np.nan
    return z


def get_maxproj(niiobj, sigma=False, biside=False):
    """
    sigma: smoothness
    biside: True if data contains negative value
    """
    data = decomp_dataobj(niiobj)[0]
    
    # prepare coordinate system
    x, y, z = get_meshgrid(niiobj)
    
    if biside:
        data_pos = data.copy()
        data_neg = data.copy()
    
        data_pos[data < 0] = 0
        data_neg[data > 0] = 0
    
        axi_img = np.max(data_pos, axis=2) + np.min(data_neg, axis=2)
        cor_img = np.max(data_pos, axis=1).T + np.min(data_neg, axis=1).T
        sag_img = np.max(data_pos, axis=0).T + np.min(data_neg, axis=0).T
    else:
        axi_img = np.max(data, axis=2)
        cor_img = np.max(data, axis=1).T
        sag_img = np.max(data, axis=0).T
    if sigma:
        axi_img = blur(axi_img, sigma=sigma)
        cor_img = blur(cor_img, sigma=sigma)
        sag_img = blur(sag_img, sigma=sigma)
    else:
        axi_img[axi_img == 0] = np.nan
        cor_img[cor_img == 0] = np.nan
        sag_img[sag_img == 0] = np.nan
    
    return dict(axial=(y, x, axi_img),
                coronal=(x, z, cor_img),
                sagittal=(y, z, sag_img))


def orthoview_maxproj(anat_obj, sigma=False, **kwargs):
    data, affine, resol = decomp_dataobj(anat_obj)
    
    # parse parameters
    figsize = kwargs['figsize']        if 'figsize'        in kwargs.keys() else None
    dpi     = kwargs['dpi']            if 'dpi'            in kwargs.keys() else None
    cmap    = kwargs['cmap']           if 'cmap'           in kwargs.keys() else 'Greys'
    cmapsf  = kwargs['cmap_max_scale'] if 'cmap_max_scale' in kwargs.keys() else 1
    alpha   = kwargs['alpha']          if 'alpha'          in kwargs.keys() else 0.2
    vmin    = kwargs['vmin']           if 'vmin'           in kwargs.keys() else 0
    vmax    = kwargs['vmax']           if 'vmax'           in kwargs.keys() else data.max() * cmapsf
    
    # prepare coordinate system
    size = data.shape
    mpovj = get_maxproj(anat_obj, sigma=sigma, biside=False)
    
    ax_idx = [(0, 0), (0, 1), (1, 0), (1, 1)]

    input_set = [None, mpovj['axial'], mpovj['coronal'], mpovj['sagittal']]
    m = size[0] + size[1]
    n = size[2] + size[0]
    
    if figsize is None:
        if m > n:
            denom = n
        else:
            denom = m
        figsize=(m/denom, n/denom)
    
    # figsize, scale correction
    figsize = (np.array(list(figsize)) * 3).tolist()
    fig, ax = plt.subplots(2, 2, figsize=figsize, dpi=dpi,
                           gridspec_kw={'width_ratios': [size[0], size[1]],
                                        'height_ratios': [size[0], size[2]],
                                        'wspace': 0, 'hspace': 0}, constrained_layout=True)
    for idx, (i, j) in enumerate(ax_idx):
        if idx == 0:
            # remove axis
            ax[i, j].axis('off')
        else:
            h, w, img = input_set[idx]
            ax[i, j].pcolormesh(h, w, img, 
                                cmap=cmap, alpha=alpha,
                                vmin=vmin, vmax=vmax)
            ax[i, j].set_aspect(1)
            ax[i, j].grid(False)
            ax[i, j].axis('off')
            if idx in [1, 3]:
                ax[i, j].invert_xaxis()
    return fig

def orthoview_maxproj_overlay(func_obj, fig, **kwargs):
    mpovj = get_maxproj(func_obj, sigma=False, biside=True)
    
    # prepare figure
    axes = fig.axes
    input_set = [None, mpovj['axial'], mpovj['coronal'], mpovj['sagittal']]
    
    for idx, ax in enumerate(axes):
        if idx == 0:
            # remove axis
            ax.axis('off')
        else:
            h, w, img = input_set[idx]
            ax.pcolormesh(h, w, img, edgecolor=None, **kwargs)
    return fig

def orthoview_slice(anat_obj, func_obj, coord, **kwargs):
    import matplotlib.lines as mlines
    offsets = dict(axial=[0, 0], coronal=[0, 0], sagittal=[0, 0])
    
    # parse parameters
    figsize   = kwargs['figsize']        if 'figsize'   in kwargs.keys() else None
    dpi       = kwargs['dpi']            if 'dpi'       in kwargs.keys() else None
    vmin      = kwargs['vmin']           if 'vmin'      in kwargs.keys() else -5
    vmax      = kwargs['vmax']           if 'vmax'      in kwargs.keys() else 5
    v_offsets = kwargs['v_offsets']      if 'v_offsets' in kwargs.keys() else offsets
    h_offsets = kwargs['h_offsets']      if 'h_offsets' in kwargs.keys() else offsets
    cmap      = kwargs['cmap']           if 'cmap'      in kwargs.keys() else 'coolwarm'
    
    for k in ['figsize', 'dpi', 'vmin', 'vmax', 'v_offsets', 'h_offsets', 'cmap']:
        if k in kwargs.keys():
            del kwargs[k]
    
    fig = orthoview_maxproj(anat_obj, cmap='gray', sigma=2, figsize=figsize, dpi=dpi,
                            alpha=0.5, vmin=-50000, vmax=50000)
    
    anat_affine = decomp_dataobj(anat_obj)[1]
    anat_coord = mm_to_voxel(coord, anat_affine)
    anat_slice_obj = get_slice(anat_obj, anat_coord)
   
    if func_obj is not None:
        func_affine = decomp_dataobj(func_obj)[1]
        func_coord = mm_to_voxel(coord, func_affine)
        func_slice_obj = get_slice(func_obj, func_coord)
        func_set = [None, func_slice_obj['axial'], func_slice_obj['coronal'], func_slice_obj['sagittal']]
     
    axes = fig.axes
    anat_set = [None, anat_slice_obj['axial'], anat_slice_obj['coronal'], anat_slice_obj['sagittal']]
    v_offset = [None, v_offsets['axial'], v_offsets['coronal'], v_offsets['sagittal']]
    h_offset = [None, h_offsets['axial'], h_offsets['coronal'], h_offsets['sagittal']]
    line_set = [None, (coord[1], coord[0]), (coord[0], coord[2]), (coord[1],coord[2])]    
    
    for idx, ax in enumerate(axes):
        if idx == 0:
            # remove axis
            ax.axis('off')
        else:
            ah, aw, anat_img = anat_set[idx]
            lh, lw = line_set[idx]
            v_off = v_offset[idx]
            h_off = h_offset[idx]
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            ax.pcolormesh(ah, aw, anat_img, cmap='binary', vmin=-30000, vmax=70000, **kwargs)
            ax.pcolormesh(ah, aw, anat_img, cmap='binary', vmin=7000, vmax=30000, **kwargs)
            ax.pcolormesh(ah, aw, get_outline(anat_img), cmap='Greys_r', alpha=0.3, vmax=3)
            if func_obj is not None:
                fh, fw, func_img = func_set[idx]
                ax.pcolormesh(fh, fw, func_img, cmap=cmap, alpha=1, vmin=vmin, vmax=vmax)
            l1 = mlines.Line2D([lh, lh], [ymin - v_off[0], ymax - v_off[1]], 
                               linestyle=':', linewidth=1, color='black')
            l2 = mlines.Line2D([xmin - h_off[0], xmax - h_off[1]], [lw, lw], 
                               linestyle=':', linewidth=1, color='black')
            ax.add_line(l1)
            ax.add_line(l2)
    return fig


def orthoview_paxinos(anat_obj, func_obj, coord, vmin=-5, vmax=5, cmap='coolwarm'):
    matrix_coord = paxinose_to_camri(*coord)
    fig = orthoview_slice(anat_obj, func_obj, matrix_coord, dpi=100,
                          v_offsets=dict(axial=[-2, 2], coronal=[-2, 0], sagittal=[-2, 0]),
                          vmin=vmin, vmax=vmax, cmap=cmap)

    label = ['LR', 'PA', 'SI'][::-1]
    coord = coord[::-1]
    for i, ax in enumerate(fig.get_axes()):
        if i != 0:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.text(xmax, ymin, f'{label[i-1]}: {coord[i-1]:.2f} mm', 
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=8)
    return fig

def mosaicview_slice(anat_obj, func_obj, slice_coords, view='coronal', vmin=-5, vmax=5, navi=True
                     ,**kwargs):
    
    cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else 'coolwarm'
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 1
    interpolation = kwargs['interpolation'] if 'interpolation' in kwargs.keys() else 'nearest'
    
    
    """ Incompleted code """
    import matplotlib.lines as mlines
    mpobj = get_maxproj(anat_obj)
    ah, am, aimg = mpobj[view]

    # nevigation axis
    nav_ref = dict(axial='sagittal', coronal='sagittal', sagittal='axial')
    slc_ref = dict(axial=np.array([0, 0, 1]),
                   coronal=np.array([0, 1, 0]),
                   sagittal=np.array([1, 0, 0]))

    anat_coords = [mm_to_voxel(slc_ref[view] * s, anat_obj.affine) for s in slice_coords]
    func_coords = [mm_to_voxel(slc_ref[view] * s, func_obj.affine) for s in slice_coords]

    mpobj = get_maxproj(anat_obj, sigma=2)
    slobj = get_slice(anat_obj, coord=mm_to_voxel([0, 0, 0], anat_obj.affine))
    _, _, rimg = mpobj[view]
    nh, nw, nimg = mpobj[nav_ref[view]]
    sh, sw, simg = slobj[nav_ref[view]]

    n_slices = len(slice_coords)
    n = rimg.shape[-1]
    
    if navi:
        m = rimg.shape[0] * n_slices + rimg.shape[1]
    else:
        m = rimg.shape[0] * n_slices
        
    if m < n:
        figsize = [m/m, n/m]
    else:
        figsize = [m/n, n/n]
        
    figsize = np.array(figsize) * 3

    fig = plt.figure(constrained_layout=False, dpi=150, figsize=figsize)
    
    if navi:
        gs = fig.add_gridspec(nrows=1, ncols=n_slices + 1, 
                              left=0, right=1, wspace=0, hspace=0, 
                              width_ratios=[rimg.shape[1]/rimg.shape[0]] + [1]*n_slices,
                              height_ratios=[1])

        nev_ax = fig.add_subplot(gs[0, 0])
        nev_ax.pcolormesh(nh, nw, nimg, cmap='Greys', vmin=-10000, vmax=60000)
        nev_ax.pcolormesh(sh, sw, simg, cmap='binary', vmin=-30000, vmax=80000, shading='gouraud')
        nev_ax.pcolormesh(nh, nw, get_outline(simg), cmap='binary', alpha=1, vmin=0, vmax=1.8)
        nev_ax.set_aspect(1)
        nev_ax.invert_xaxis()
        nev_ax.axis('off')
        ymin, ymax = nev_ax.get_ylim()
        for sc in slice_coords:
            l = mlines.Line2D([sc, sc], [ymin, ymax],
                              linestyle=':', linewidth=1, color='black')
            nev_ax.add_line(l)
    else:
        gs = fig.add_gridspec(nrows=1, ncols=n_slices, 
                              left=0, right=1, wspace=0, hspace=0, 
                              width_ratios=[1]*n_slices)
    for i, s in enumerate(anat_coords):
        # slice axis
        ax = fig.add_subplot(gs[0, i + navi])
        slc_img = get_slice(anat_obj, coord=s)
        fnc_img = get_slice(func_obj, coord=func_coords[i])
        anat_h, anat_w, anat_img = slc_img[view]
        func_h, func_w, func_img = fnc_img[view]
        func_img[func_img == 0] = np.nan
        ax.pcolormesh(anat_h, anat_w, anat_img, cmap='Greys', vmin=7000, vmax=30000, shading='gouraud')
        ax.pcolormesh(anat_h, anat_w, get_outline(anat_img), cmap='binary', alpha=1, vmin=0, vmax=1.8)
        ax.pcolormesh(func_h, func_w, func_img, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax.set_aspect(1)
        ax.axis('off')
    return fig

def mosaicview_paxinos(anat_obj, func_obj, coord, navi=True, annotate=True, 
                       vmin=-5, vmax=5, cmap='coolwarm', fontsize=15, view='coronal', **kwargs):
    slice_coords = np.array(coord) + 0.36
    fig = mosaicview_slice(anat_obj, func_obj, slice_coords, view=view, vmin=vmin, vmax=vmax, 
                           navi=navi, cmap=cmap, **kwargs)

    if annotate:
        for i, ax in enumerate(fig.get_axes()):
            if navi:
                if i != 0:
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    ax.text((xmin + xmax)/2, ymin - 2, f'{coord[i-1]:.2f} mm', 
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=fontsize)
            else:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.text((xmin + xmax)/2, ymin - 2, f'{coord[i]:.2f} mm', 
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=fontsize)
    return fig
                   