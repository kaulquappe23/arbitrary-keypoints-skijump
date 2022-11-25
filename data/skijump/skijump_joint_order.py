
yt_skijump_reference_joints = [1, 10]


class YTSkijumpJointOrder:
    head = 0
    r_shoulder = 1
    r_elbow = 2
    r_hand = 3
    l_shoulder = 4
    l_elbow = 5
    l_hand = 6
    r_hip = 7
    r_knee = 8
    r_ankle = 9
    l_hip = 10
    l_knee = 11
    l_ankle = 12
    r_skitip = 13
    r_skitail = 14
    l_skitip = 15
    l_skitail = 16

    num_joints = 17
    num_bodyparts = 8

    @classmethod
    def indices(cls):
        return [cls.head,
                cls.r_shoulder, cls.r_elbow, cls.r_hand,
                cls.l_shoulder, cls.l_elbow, cls.l_hand,
                cls.r_hip, cls.r_knee, cls.r_ankle,
                cls.l_hip, cls.l_knee, cls.l_ankle,
                cls.r_skitip, cls.r_skitail, cls.l_skitip, cls.l_skitail]

    @classmethod
    def line_bodypart_indices(cls):
        return [[cls.r_shoulder, cls.r_elbow], [cls.r_elbow, cls.r_hand],
                [cls.r_hip, cls.r_knee], [cls.r_knee, cls.r_ankle],
                [cls.l_shoulder, cls.l_elbow], [cls.l_elbow, cls.l_hand],
                [cls.l_hip, cls.l_knee], [cls.l_knee, cls.l_ankle]
                ]

    @classmethod
    def names(cls):
        return ["head",
                "rsho", "relb", "rhan",
                "lsho", "lelb", "lhan",
                "rhip", "rkne", "rank",
                "lhip", "lkne", "lank",
                "rsti", "rsta", "lsti", "lsta"]

    def __init__(self):
        pass
