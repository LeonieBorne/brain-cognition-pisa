import os
import pandas as pd

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
