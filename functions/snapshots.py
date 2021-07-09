import os
import pandas as pd

# everything should be in python 2 to work!

def view_sulcus_scores(dict_scores, side='left',
                        snapshot=None, reg_q=[0.5, -0.5, -0.5, 0.5],
                        palette='Blue-Red-fusion', minVal=None, maxVal=None,
                        background=None):
    '''
    Return an Anatomist window containing the SPAM model representation
    with a different score attributed to each sulcus

    Parameters
    ----------

    dict_scores : dict
        Dictionary containing the score attributed to each sulcus.
        E.g. dict={'S.C._left': 0.1, 'S.C._right': -0.6, etc.}

    side : str {'left', 'right'} (default 'left')
        Hemisphere side to represent

    snapshot : None or str (default None)
        If not None, a snapshot is saved at the path indicated

    reg_q : float vector, size 4, normed (default [0.5,-0.5,-0.5,0.5])
        View rotation of the snapshot

    palette : Anatomist.APalette or str (defauly 'Blue-Red-fusion')
        Principal palette to apply

    minVal : float (default 0)
        Minimum object texture value mapped to the lower bound of the
        palette, by default in relative proportional mode.

    maxVal : float (default 0)
        Maximum object texture value mapped to the lower bound of the
        palette.

    background : float vector, size 4
        Set background color. For black, use [0, 0, 0, 1].
    '''

    try:
        import anatomist.api as ana
        from soma import aims
        import sigraph

        a = ana.Anatomist()

        aims_path = aims.carto.Paths.resourceSearchPath()[-1]
        build_path = '/home/borne/brainvisa/build'
        nomenclature_filename = os.path.join(
                aims_path, 'nomenclature/hierarchy/sulcal_root_colors.hie')
        nom = aims.read(nomenclature_filename)
        anom = a.toAObject(nom)
        si = 'L' if side == 'left' else 'R'

        # mesh
        seg_path = os.path.join(
            build_path, 'bug_fix/share/brainvisa-share-4.6/models/models_2008',
            'descriptive_models/segments')
        mesh_file = os.path.join(
            seg_path,
            'global_registered_spam_%s/meshes/%swhite_spam.gii' % (side, si))
        mesh = aims.read(mesh_file)
        amesh = a.toAObject(mesh)
        ref = a.createReferential()
        amesh.assignReferential(ref)
        a.loadTransformation(os.path.join(
            seg_path,
            'global_registered_spam_left/meshes/Lwhite_TO_global_spam.trm'),
            ref, a.centralReferential())
        # graph
        graph = aims.read(os.path.join(
            seg_path,
            'global_registered_spam_{}/meshes/{}spam_model_meshes_0.arg'.format(side, si)))
        flt = sigraph.FoldLabelsTranslator()
        flt.readLabels(os.path.join(
            build_path, 'bug_fix/share/brainvisa-share-4.6/nomenclature',
            'translation/sulci_model_2018.trl'))
        flt.translate(graph)
        if minVal is None:
            nan_error = 1.01 * min(dict_scores.values()) \
                - 0.01 * max(dict_scores.values())
        else:
            nan_error = 1.01 * minVal - 0.01 * maxVal
        for vert in graph.vertices():
            if 'name' in vert:
                name = vert['name']
                if name != 'brain_hull':
                    if name in dict_scores.keys():
                        vert['error'] = dict_scores[name]
                    else:
                        vert['error'] = nan_error
                        print('%s is nan, set to 0 instead' % name)
        agraph = a.toAObject(graph)
        a.execute('GraphDisplayProperties', objects=[agraph],
                  display_mode='PropertyMap', display_property='error')
        minVal = minVal if minVal is not None else min(dict_scores.values())
        maxVal = maxVal if maxVal is not None else max(dict_scores.values())
        agraph.setPalette(palette, minVal=minVal, maxVal=maxVal,
                          absoluteMode=True, zeroCentered1=True)

        # view
        win = a.createWindow('3D')
        win.addObjects([agraph, amesh])
        a.execute('SelectByNomenclature', nomenclature=anom, names='unknown')
        sf = ana.cpp.SelectFactory.factory()
        win.removeObjects(list(sf.selected()[0]))
        win.windowConfig(cursor_visibility=0)
        win.camera(view_quaternion=reg_q)
        if background is not None:
            a.execute('WindowConfig', windows=[win],
                      light={'background': background})

        # snapshot
        if snapshot is not None:
            win.snapshot(snapshot, width=2000, height=1500)

        return win

    except:
        if snapshot is not None:
            # dict_scores
            df = pd.DataFrame(columns=['score'])
            for k, v in dict_scores.items():
                df.loc[k] = v
            # reg_q
            if reg_q == [0.5, 0.5, 0.5, 0.5]:
                rotation = 1
            else:
                rotation = 0
            # background
            if background == [0, 0, 0, 1]:
                back = 'black'
            else:
                back = 'white'

            snapshot_path = snapshot[:snapshot.rfind('/')]
            snapshot_name = snapshot[snapshot.rfind('/'):snapshot.rfind('.')]
            df.to_csv('/tmp/{}.csv'.format(snapshot_name), float_format='%.100f')
            cmd = 'XSOCK=/tmp/.X11-unix '
            cmd += '&& XAUTH=/tmp/.docker.xauth '
            cmd += "&& xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - "
            cmd += '&& docker run --rm -v /tmp/{}.csv:/home/{}.csv: '.format(snapshot_name, snapshot_name) +\
                    ' -v {}:/home/snapshot: -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH '.format(snapshot_path) +\
                    '9051257e588b /bin/bash -c ' +\
                    '". /home/brainvisa/build/bug_fix/bin/bv_env.sh /home/brainvisa/build/bug_fix ' +\
                    '&& python ./snapshots.py -c /home/{}.csv -o /home/snapshot{}.png '.format(snapshot_name, snapshot_name) +\
                    "-s {} -r {} -p '{}' -b {}".format(side, rotation, palette, back)
            if minVal is not None:
                cmd += ' -m {}'.format(minVal)
            if maxVal is not None:
                cmd += ' -n {}'.format(maxVal)
            cmd += '"'
            print(cmd)
            print()
            os.system(cmd)
        else:
            print('WARNING: PyAims not found.',
                  'Specify "snapshot" to run the function in a docker image.')

def view_sulcus_graph(gfile, mfile=None, snapshot=None,
                      reg_q=[0.5, -0.5, -0.5, 0.5], background=None):
    try:
        import anatomist.api as ana
        from soma import aims
        import sigraph

        a = ana.Anatomist()
        aims_path = aims.carto.Paths.resourceSearchPath()[-1]
        
        nomenclature_filename = os.path.join(
            aims_path, 'nomenclature/hierarchy/sulcal_root_colors.hie')
        nomenclature = aims.read(nomenclature_filename)
        anom = a.toAObject(nomenclature)

        if os.path.exists(gfile):
            graph = aims.read(gfile)
            agraph = a.toAObject(graph)
            if mfile is not None:
                mesh = aims.read(mfile)
            amesh = a.toAObject(mesh)
            win = a.createWindow('3D', options={'hidden': True})
            win.addObjects([agraph, amesh])
            win.windowConfig(cursor_visibility=0)
            win.camera(view_quaternion=reg_q)
            if background is not None:
                a.execute('WindowConfig', windows=[win],
                          light={'background': background})
            if snapshot is not None:
                win.snapshot(snapshot, width=2000, height=1500)
            # del graph
            # agraph.releaseAppRef()
            # amesh.releaseAppRef()
            return win
        else:
            print(gfile, 'does not exists.')
    except:
        if snapshot is not None:
            # python script
            f = open("/tmp/python_script.py", "w+")
            f.write("from snapshots import view_sulcus_graph \n")
            f.write("view_sulcus_graph('/home/all{}',".format(gfile))
            f.write(" '/home/all{}', '/home/all{}',".format(mfile, snapshot))
            f.write("{}, {}) \n".format(reg_q, background))
            f.close()
            
            docker_cmd()
        else:
            print('WARNING: PyAims not found.',
                  'Specify "snapshot" to run the function in a docker image.')


def view_point_score(points, points_val, vs,
                     snapshot=None, reg_q=[0.5, -0.5, -0.5, 0.5],
                     palette='Blue-Red-fusion', minVal=None, maxVal=None,
                     add_files=[], trans_tal=None):
    ''' points_val need to be positive!'''

    try:
        import anatomist.api as ana
        from soma import aims, aimsalgo
        import numpy as np
        import shutil
    
        a = ana.Anatomist()
    
        # construct vol with confiance degree
        bucket = np.asarray([list(x) for x in points])
        bb = [[min(bucket[:, 0]), max(bucket[:, 0])],
              [min(bucket[:, 1]), max(bucket[:, 1])],
              [min(bucket[:, 2]), max(bucket[:, 2])]]
    
        vol = aims.Volume_FLOAT(int(round(bb[0][1]-bb[0][0]))+1,
                                int(round(bb[1][1]-bb[1][0]))+1,
                                int(round(bb[2][1]-bb[2][0]))+1)
    
        vol.header()['translation'] = [-int(round(bb[0][0]))*vs[0],
                                       -int(round(bb[1][0]))*vs[1],
                                       -int(round(bb[2][0]))*vs[2]]
    
        trans_vx_vol = [-int(round(bb[0][0])),
                        -int(round(bb[1][0])),
                        -int(round(bb[2][0]))]
    
        vol.header()['voxel_size'] = vs
    
        np.asarray(vol)[np.asarray(vol) == 0] = -2
    
        for point, point_val in zip(points, points_val):
            point_tmp = np.asarray(point)+trans_vx_vol
            vol[point_tmp[0], point_tmp[1], point_tmp[2]] = point_val
    
        # threshold the vol to compute the mesh
        vol_threshold = aims.Volume(vol)
        np.asarray(vol_threshold)[np.asarray(vol_threshold) != -2] = 32767
        # convert to Volume_16
        c = aims.Converter(intype=vol_threshold, outtype=aims.Volume_S16)
        vol_16 = c(vol_threshold)
        # put a border -1 to avoid testing
        vol_tmp = aims.Volume_S16(vol_16.getSizeX() + 2,
                                  vol_16.getSizeY() + 2,
                                  vol_16.getSizeZ() + 2)
        vol_tmp.copyHeaderFrom(vol_16.header())
        vol_border = aims.VolumeView(vol_tmp, [1, 1, 1], vol_16.getSize())
        np.asarray(vol_tmp)[:, :, :, 0] = -1
        np.asarray(vol_border)[:, :, :, :] = np.asarray(vol_16)
    
        # create a directory to store the meshes
        if os.path.isdir('/tmp/meshes'):
            shutil.rmtree('/tmp/meshes')
            os.makedirs('/tmp/meshes')
        else:
            os.mkdir('/tmp/meshes')
        # construct the mesh
        m = aimsalgo.Mesher()
        vol_border.header()['voxel_size'] = vs
        m.doit(vol_border, '/tmp/meshes/mesh.gii')
    
        # load the meshes & fusion vol and meshes
        fus_list = []
        avol = a.toAObject(vol)
        minVal = min(points_val) if minVal is None else minVal
        maxVal = max(points_val) if maxVal is None else maxVal
        avol.setPalette(palette, minVal=minVal, maxVal=maxVal, absoluteMode=True)
        trans_vol = vol.header()['translation']
        ref_m = a.createReferential()
        a.execute('LoadTransformation', origin=ref_m,
                  destination=a.centralReferential(),
                  matrix=[-trans_vol[0], -trans_vol[1], -trans_vol[2],
                          1, 0, 0,  0, 1, 0,  0, 0, 1])
        for file in os.listdir('/tmp/meshes'):
            if file.endswith('.gii'):
                amesh = a.loadObject(os.path.join('/tmp/meshes', file))
                # fusion
                fus = a.fusionObjects([avol, amesh], method='Fusion3DMethod')
                a.execute('Fusion3DParams', object=fus, method='line_internal',
                           submethod='max', depth=1.2, step=0.3)
                # referential
                avol.assignReferential(ref_m)
                amesh.assignReferential(ref_m)
                # save objects
                fus_list.append(fus)

        # ref_nat = a.createReferential()
        # a.execute('LoadTransformation', origin=ref_nat,
        #           destination=a.centralReferential(),
        #           matrix=[trans_tal[0][3], trans_tal[1][3], trans_tal[2][3],
        #                   trans_tal[0][0], trans_tal[0][1], trans_tal[0][2],
        #                   trans_tal[0][1], trans_tal[1][1], trans_tal[1][2],
        #                   trans_tal[0][2], trans_tal[2][1], trans_tal[2][2]])
        for file in add_files:
            aobj = a.loadObject(file)
            # aobj.assignReferential(ref_nat)
            fus_list.append(aobj)

        win = a.createWindow('3D')
        win.addObjects(fus_list)
        win.windowConfig(cursor_visibility=0)
        win.camera(view_quaternion=reg_q)
            
        # snapshot
        if snapshot is not None:
            win.snapshot(snapshot, width=2000, height=1500)

        return win, fus_list

    except:
        if snapshot is not None:
            # python script
            f = open("/tmp/python_script.py", "w+")
            f.write("from snapshots import view_point_score \n")
            f.write("view_point_score({},".format([list(p) for p in points]))
            f.write(" {}, {},".format(points_val, vs))
            f.write("'/home/all{}', {},".format(snapshot, reg_q))
            f.write(" '{}', {}, {}".format(palette, minVal, maxVal))
            f.write(", {}".format(['/home/all{}'.format(af) for af in add_files]))
            f.write(", {}) \n".format(trans_tal))
            f.close()
            
            docker_cmd()
        else:
            print('WARNING: PyAims not found.',
                  'Specify "snapshot" to run the function in a docker image.')

def docker_cmd():
    cmd = 'XSOCK=/tmp/.X11-unix '
    cmd += '&& XAUTH=/tmp/.docker.xauth '
    cmd += "&& xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - "
    cmd += '&& docker run --rm -v /:/home/all: ' +\
           ' -v /home/leonie/Documents/source/hmri/python/librairies/uon/snapshots.py:/home/snapshots.py: ' +\
           ' -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH ' +\
           ' -v /tmp/python_script.py:/home/python_script.py: ' +\
           '9051257e588b /bin/bash -c ' +\
           '". /home/brainvisa/build/bug_fix/bin/bv_env.sh /home/brainvisa/build/bug_fix ' +\
           '&& export QT_X11_NO_MITSHM=1 && python /home/python_script.py "'
    print(cmd)
    print()
    os.system(cmd)    
