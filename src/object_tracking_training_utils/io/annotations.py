import xml.etree.ElementTree as ET
from os.path import join, isdir, isfile, splitext

def get_annotations(annot_dir, sequence_dir, frame_file):
    """ Gets the annotations contained in the xml file of the current frame.

    Args:
        annot_dir (str): The root dir of annotation folders
        sequence_dir(str): The directory in which frame is located, relative to
        root directory
        frame_file (str): The frame filename
        i.e., sequence_identifier/frame_number.JPEG, or the full path.
    Return:
        dictionary: A dictionary containing the four values of the
            the bounding box stored in the following keywords:
            'xmax' (int): The rightmost x coordinate of the bounding box.
            'xmin' (int): The leftmost x coordinate of the bounding box.
            'ymax' (int): The uppermost y coordinate of the bounding box.
            'ymin' (int): The lowest y coordinate of the bounding box.

        width (int): The width of the jpeg image in pixels.
        height (int): The height of the jpeg image in pixels.
        valid_frame (bool): False if the target is present in the frame,
            and True otherwise.
        OBS: If the target is not in the frame the bounding box information
        are all None.
    """
    frame_number = splitext(frame_file)[0]
    annot_path = join(annot_dir, sequence_dir, frame_number + '.xml')
    if isfile(annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if root.find('object') is None:
            annotation = {'xmax': None, 'xmin': None, 'ymax': None, 'ymin': None}
            valid_frame = False
            return annotation, width, height, valid_frame
        # TODO Decide what to do with multi-object sequences. For now I'm
        # just going to take the one with track_id = 0
        track_id_zero_present = False
        for obj in root.findall('object'):
            if obj.find('trackid').text == '0':
                bbox = obj.find('bndbox')
                xmax = int(bbox.find('xmax').text)
                xmin = int(bbox.find('xmin').text)
                ymax = int(bbox.find('ymax').text)
                ymin = int(bbox.find('ymin').text)
                track_id_zero_present = True
        if not track_id_zero_present:
            annotation = {'xmax': None, 'xmin': None, 'ymax': None, 'ymin': None}
            valid_frame = False
            return annotation, width, height, valid_frame
    else:
        raise FileNotFoundError("The file {} could not be found"
                                .format(annot_path))

    annotation = {'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}
    valid_frame = True
    return annotation, width, height, valid_frame


def check_folder_tree(root_dir):
    """ Checks if the folder structure is compatible with the expected one.

    Args:
        root_dir (str): The path to the root directory of the dataset.
    Return:
        bool: True if the structure is compatible, False otherwise.
    """
    data_type = ['Annotations', 'Data']
    dataset_type = ['train', 'val']
    necessary_folders = [join(root_dir, data, 'VID', dataset) for data in data_type for dataset in dataset_type]
    return all(isdir(path) for path in necessary_folders)

