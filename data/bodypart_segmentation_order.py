from data.skijump.skijump_joint_order import YTSkijumpJointOrder


class YTSkijumpBodypartOrder:
    head = 1
    torso = 2
    l_uarm = 3
    r_uarm = 4
    l_farm = 5
    r_farm = 6
    l_hand = 7
    r_hand = 8
    l_thigh = 9
    r_thigh = 10
    l_lleg = 11
    r_lleg = 12
    l_foot = 13
    r_foot = 14
    l_ski = 15
    r_ski = 16

    @classmethod
    def get_max_bodypart_num(cls):
        return 16

    @classmethod
    def get_colors(cls):
        colors = ["black", # background
                  "white",
                  "silver",
                  "green",
                  "yellow",
                  "limegreen",
                  "orange",
                  "greenyellow",
                  "gold" ,
                  "dodgerblue",
                  "orchid",
                  "blue",
                  "red",
                  "lightskyblue",
                  "firebrick",
                  "aqua",
                  "lightsalmon",
                  ]
        return colors

    @classmethod
    def get_keypoint_bodypart_triples(cls):
        return [(YTSkijumpJointOrder.l_hand, YTSkijumpJointOrder.l_elbow, cls.l_farm),
                (YTSkijumpJointOrder.r_hand, YTSkijumpJointOrder.r_elbow, cls.r_farm),
                (YTSkijumpJointOrder.l_elbow, YTSkijumpJointOrder.l_shoulder, cls.l_uarm),
                (YTSkijumpJointOrder.r_elbow, YTSkijumpJointOrder.r_shoulder, cls.r_uarm),
                (YTSkijumpJointOrder.l_ankle, YTSkijumpJointOrder.l_knee, cls.l_lleg),
                (YTSkijumpJointOrder.r_ankle, YTSkijumpJointOrder.r_knee, cls.r_lleg),
                (YTSkijumpJointOrder.l_knee, YTSkijumpJointOrder.l_hip, cls.l_thigh),
                (YTSkijumpJointOrder.r_knee, YTSkijumpJointOrder.r_hip, cls.r_thigh),
                (YTSkijumpJointOrder.l_skitip, YTSkijumpJointOrder.l_skitail, cls.l_ski),
                (YTSkijumpJointOrder.r_skitip, YTSkijumpJointOrder.r_skitail, cls.r_ski), ]

    @classmethod
    def get_bodypart_to_keypoint_dict(cls):
        result_dict = {}
        for keypoint1, keypoint2, bodypart in cls.get_keypoint_bodypart_triples():
            result_dict[bodypart] = (keypoint1, keypoint2)
        return result_dict

    @classmethod
    def names(cls):
        return ["background", "head", "torso", "l_uarm", "r_uarm", "l_farm", "r_farm", "l_hand", "r_hand", "l_thigh", "r_thigh", "l_lleg", "r_lleg", "l_foot", "r_foot", "l_ski", "r_ski"]