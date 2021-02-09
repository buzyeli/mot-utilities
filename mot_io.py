from abc import ABC, abstractmethod


class MotFormat(ABC):
    def __init__(self, id, frame_id, initial_bbox):
        self.id = id
        self.frame_ids = [frame_id]
        self.bboxes = [initial_bbox]

    def update_state(self):
        pass

    def _get_state(self, idx):
        pass

    def get_final_state(self):
        return self._get_state(-1)

    def get_state_in_frame(self, frame_id):
        return self._get_state(self.frame_ids.index(frame_id))


class MotTargetFormat(MotFormat):
    def __init__(self, id, type, activity, frame_id, initial_bbox, visibility):
        super().__init__(id, frame_id, initial_bbox)
        self.type = type
        self.activity = activity
        self.visibility = [visibility]

    def update_state(self, frame_id, bbox, visibility):
        self.frame_ids.append(frame_id)
        self.bboxes.append(bbox)
        self.visibility.append(visibility)

    def _get_state(self, idx):
        return self.frame_ids[idx], self.bboxes[idx], self.visibility[idx]


class MotDetFormat(MotFormat):
    def __init__(self, id, frame_id, initial_bbox, conf_score):
        super().__init__(id, frame_id, initial_bbox)
        self.conf_score = [conf_score]

    def update_state(self, frame_id, bbox, conf_score):
        self.frame_ids.append(frame_id)
        self.bboxes.append(bbox)
        self.conf_score.append(conf_score)

    def _get_state(self, idx):
        return self.frame_ids[idx], self.bboxes[idx], self.conf_score[idx]


class MotGt:
    """
    Design of MotGt is based on the assumption that we know the trajectory of every unique item thanks to ground-truth.
    So instead of drawing random bboxes for each frame i decided to keep a list of unique targets.
    """
    def __init__(self, filename):
        self.targets = self._read_file(filename) # TODO: Check if converting this to a dict will improve the code in general

    def get_objects_in_frame(self, frame_id):
        targets_in_frame = []
        for target in self.targets:
            if frame_id in target.frame_ids:
                targets_in_frame.append(target)
        return targets_in_frame

    def _read_file(self, filename):
        """
        File format for MOT 2016
        [frame_id],[trajectory_id],[top_left_x],[top_left_y],[width],[height],[ignore/active],[object_type],[visibility_ratio]

        :param filename: Full path to the ground truth file
        :return: List of target objects (MotTargetFormat)
        """
        targets = []
        with open(filename,'r') as f:
            for line in f:
                tokens = line.split(',')
                tmp_tokens = [x.strip() for x in tokens]
                frame_id = int(tmp_tokens[0])
                object_id = int(tmp_tokens[1])
                top_left_x, top_left_y, width, height = int(tmp_tokens[2]), int(tmp_tokens[3]), int(tmp_tokens[4]), int(tmp_tokens[5])
                activity, type, visibility = int(tmp_tokens[6]), int(tmp_tokens[7]), float(tmp_tokens[8])

                target = next((t for t in targets if t.id == object_id), None)

                if target is None:
                    targets.append(MotTargetFormat(object_id, type, activity, frame_id, [top_left_x, top_left_y, width, height], visibility))
                else:
                    target.update_state(frame_id, [top_left_x, top_left_y, width, height], visibility)

        return targets


class MotDet:
    """
    In contrast (to MotGt) MotDet does not provide unique object ids. Instead it only keeps track of multiple bboxes for each frame.

    """
    def __init__(self, filename):
        self.bboxes = self._read_file(filename) # NOTE: This is dict, in contrast to MotGt where it was an array

    def get_objects_in_frame(self, frame_id):
        return self.bboxes[frame_id]

    def _read_file(self, filename):
        """
        File format for MOT 2016
        [frame_id],[trajectory_id (expected to be -1)],[top_left_x],[top_left_y],[width],[height],[confidence score],[ignore other parameters if exist]

        :param filename: Full path to the ground truth file
        :return: List of target objects (MotDetFormat)
        """
        bboxes_in_frames = {}
        last_frame_id = -1
        tmp_bboxes = []
        with open(filename,'r') as f:
            for line in f:
                tokens = line.split(',')
                tmp_tokens = [x.strip() for x in tokens]
                frame_id = int(tmp_tokens[0])
                object_id = int(tmp_tokens[1]) # expected to be -1
                top_left_x, top_left_y, width, height = float(tmp_tokens[2]), float(tmp_tokens[3]), float(tmp_tokens[4]), float(tmp_tokens[5])
                conf_score = float(tmp_tokens[6])

                # Whenever we finish a frame we dump all bboxes from that frame into the dict
                if (last_frame_id != frame_id) and last_frame_id != -1:
                    bboxes_in_frames.update({last_frame_id:tmp_bboxes})
                    tmp_bboxes = []

                last_frame_id = frame_id
                tmp_bboxes.append(MotDetFormat(object_id, frame_id, [top_left_x, top_left_y, width, height], conf_score))
        return bboxes_in_frames