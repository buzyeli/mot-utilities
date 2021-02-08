
class MotTarget:
    def __init__(self, id, type, activity, frame_id, initial_bbox, visibility):
        self.id = id
        self.type = type
        self.activity = activity
        self.frame_ids = [frame_id]
        self.bboxes = [initial_bbox]
        self.visibility = [visibility]

    def update_state(self, frame_id, bbox, visibility):
        self.frame_ids.append(frame_id)
        self.bboxes.append(bbox)
        self.visibility.append(visibility)

    def _get_state(self, idx):
        return self.frame_ids[idx], self.bboxes[idx], self.visibility[idx]

    def get_final_state(self):
        return self._get_state(-1)

    def get_state_in_frame(self, frame_id):
        return self._get_state(self.frame_ids.index(frame_id))


class MotGt:
    def __init__(self, filename):
        self.targets = self._read_gt_file(filename)

    def get_targets_in_frame(self, frame_id):
        targets_in_frame = []
        for target in self.targets:
            if frame_id in target.frame_ids:
                targets_in_frame.append(target)
        return targets_in_frame

    def _read_gt_file(self, filename):
        """
        File format for MOT 2016
        [frame_id],[trajectory_id],[top_left_x],[top_left_y],[width],[height],[ignore/active],[object_type],[visibility_ratio]

        :param filename: Full path to the ground truth file
        :return: List of target objects (MotTarget)
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
                    targets.append(MotTarget(object_id, type, activity, frame_id, [top_left_x, top_left_y, width, height], visibility))
                else:
                    target.update_state(frame_id, [top_left_x, top_left_y, width, height], visibility)

        return targets