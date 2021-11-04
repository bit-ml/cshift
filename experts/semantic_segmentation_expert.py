import os

import numpy as np
import torch

from experts.basic_expert import BasicExpert
from experts.semantic_segmentation.hrnet.mit_semseg.models import ModelBuilder

current_dir_name = os.path.dirname(os.path.realpath(__file__))
ss_model_path = os.path.join(current_dir_name, 'models/')


# ADE20k labels https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
# Taskonomy labels https://github.com/StanfordVL/taskonomy/blob/master/taskbank/assets/web_assets/pseudosemantics/coco_selected_classes.txt

ADE20k_2_NYU19 = np.array([
    1, 19, 19, 2, 19, 14, 19, 4, 8, 19, 3, 19, 19, 19, 7, 6, 19, 15, 11, 5, 19,
    19, 10, 4, 9, 19, 19, 10, 13, 19, 5, 5, 19, 6, 19, 3, 18, 17, 19, 12, 3,
    15, 1, 19, 3, 6, 19, 17, 19, 19, 16, 19, 19, 19, 19, 9, 6, 12, 7, 19, 19,
    19, 9, 16, 6, 17, 15, 15, 19, 19, 6, 16, 15, 6, 16, 5, 19, 6, 19, 19, 19,
    12, 18, 19, 19, 18, 19, 19, 19, 16, 19, 19, 12, 19, 19, 19, 19, 12, 15, 6,
    10, 19, 19, 19, 19, 19, 19, 16, 15, 19, 5, 19, 15, 19, 19, 15, 19, 19, 16,
    15, 15, 19, 19, 19, 16, 15, 19, 19, 19, 16, 16, 12, 15, 16, 18, 15, 19, 19,
    19, 16, 19, 16, 15, 16, 19, 17, 16, 15, 15, 19
])
NYU40_2_NYU19 = np.array([
    1, 2, 3, 4, 5, 4, 6, 7, 8, 9, 10, 6, 11, 6, 9, 11, 3, 12, 10, 13, 12, 14,
    15, 16, 16, 15, 12, 11, 15, 10, 19, 3, 17, 17, 18, 17, 15, 19, 19, 19
])

all_classes_occurences_ade = [
    (0.0, 549870074), (5.0, 196085021), (3.0, 130539277), (14.0, 57923488),
    (8.0, 41506029), (10.0, 38603593), (23.0, 18925060), (38.0, 15416607),
    (15.0, 15124148), (7.0, 12234036), (22.0, 11429933), (19.0, 10943232),
    (1.0, 6290361), (30.0, 6262136), (24.0, 5909215), (18.0, 5713509),
    (59.0, 5369725), (28.0, 4721785), (4.0, 3909738), (27.0, 3280680),
    (42.0, 2701446), (39.0, 2670298), (49.0, 2570622), (45.0, 2492901),
    (37.0, 2218113), (2.0, 2081942), (35.0, 1996099), (36.0, 1960913),
    (50.0, 1951255), (57.0, 1630938), (58.0, 1622800), (53.0, 1536057),
    (47.0, 1391282), (64.0, 1332222), (44.0, 1283270), (63.0, 1247191),
    (73.0, 1212990), (70.0, 1202752), (17.0, 1186359), (71.0, 1001847),
    (85.0, 874339), (31.0, 802744), (69.0, 753764), (121.0, 709733),
    (41.0, 667199), (11.0, 662240), (13.0, 634054), (82.0, 617574),
    (135.0, 570853), (33.0, 568563), (81.0, 535881), (118.0, 534498),
    (139.0, 484158), (124.0, 483574), (97.0, 458518), (89.0, 452971),
    (55.0, 409462), (110.0, 395923), (133.0, 317379), (92.0, 304844),
    (65.0, 292728), (117.0, 270281), (95.0, 269361), (146.0, 264597),
    (6.0, 242926), (66.0, 239119), (100.0, 224468), (67.0, 223510),
    (21.0, 188272), (40.0, 182143), (142.0, 170561), (77.0, 164337),
    (62.0, 143480), (107.0, 136812), (9.0, 130082), (148.0, 128082),
    (112.0, 127847), (141.0, 118195), (129.0, 117848), (147.0, 117666),
    (132.0, 104225), (125.0, 101879), (32.0, 101275), (137.0, 100382),
    (131.0, 83629),
    (25.0, 82969), (75.0, 66511), (134.0, 65855), (43.0, 61356), (74.0, 54641),
    (143.0, 49683), (94.0, 48872), (138.0, 48028),
    (99.0, 46676), (144.0, 46411), (98.0, 31416), (26.0, 29338), (16.0, 29163),
    (145.0, 27637), (108.0, 23065), (130.0, 22452), (93.0, 21959),
    (46.0, 19635), (127.0, 16455), (34.0, 15702), (56.0, 14740), (54.0, 13433),
    (115.0, 12659), (61.0, 11729), (120.0, 11551), (20.0, 9595), (122.0, 8760),
    (48.0, 8723), (12.0, 8223), (76.0, 7566), (52.0, 5115), (90.0, 4067),
    (84.0, 3440), (87.0, 2077), (104.0, 2034), (72.0, 1985), (51.0, 1668),
    (86.0, 1128), (105.0, 942), (106.0, 841), (111.0, 507), (116.0, 420),
    (102.0, 382), (119.0, 360), (88.0, 300), (114.0, 262), (83.0, 260),
    (123.0, 153), (101.0, 132), (126.0, 124), (149.0, 78), (78.0, 46),
    (60.0, 35), (29.0, 33), (140.0, 30), (96.0, 15), (103.0, 3)
]

LABELS_NAME = [
    'wall', 'building;edifice', 'sky', 'floor;flooring', 'tree', 'ceiling',
    'road;route', 'bed', 'windowpane;window', 'grass', 'cabinet',
    'sidewalk;pavement', 'person;individual;someone;somebody;mortal;soul',
    'earth;ground', 'door;double;door', 'table', 'mountain;mount',
    'plant;flora;plant;life', 'curtain;drape;drapery;mantle;pall', 'chair',
    'car;auto;automobile;machine;motorcar', 'water', 'painting;picture',
    'sofa;couch;lounge', 'shelf', 'house', 'sea', 'mirror',
    'rug;carpet;carpeting', 'field', 'armchair', 'seat', 'fence;fencing',
    'desk', 'rock;stone', 'wardrobe;closet;press', 'lamp',
    'bathtub;bathing;tub;bath;tub', 'railing;rail', 'cushion',
    'base;pedestal;stand', 'box', 'column;pillar', 'signboard;sign',
    'chest;of;drawers;chest;bureau;dresser', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace;hearth;open;fireplace', 'refrigerator;icebox',
    'grandstand;covered;stand', 'path', 'stairs;steps', 'runway',
    'case;display;case;showcase;vitrine',
    'pool;table;billiard;table;snooker;table', 'pillow', 'screen;door;screen',
    'stairway;staircase', 'river', 'bridge;span', 'bookcase', 'blind;screen',
    'coffee;table;cocktail;table',
    'toilet;can;commode;crapper;pot;potty;stool;throne', 'flower', 'book',
    'hill', 'bench', 'countertop',
    'stove;kitchen;stove;range;kitchen;range;cooking;stove', 'palm;palm;tree',
    'kitchen;island',
    'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system',
    'swivel;chair', 'boat', 'bar', 'arcade;machine',
    'hovel;hut;hutch;shack;shanty',
    'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle',
    'towel', 'light;light;source', 'truck;motortruck', 'tower',
    'chandelier;pendant;pendent', 'awning;sunshade;sunblind',
    'streetlight;street;lamp', 'booth;cubicle;stall;kiosk',
    'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box',
    'airplane;aeroplane;plane', 'dirt;track',
    'apparel;wearing;apparel;dress;clothes', 'pole', 'land;ground;soil',
    'bannister;banister;balustrade;balusters;handrail',
    'escalator;moving;staircase;moving;stairway',
    'ottoman;pouf;pouffe;puff;hassock', 'bottle', 'buffet;counter;sideboard',
    'poster;posting;placard;notice;bill;card', 'stage', 'van', 'ship',
    'fountain', 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter',
    'canopy', 'washer;automatic;washer;washing;machine', 'plaything;toy',
    'swimming;pool;swimming;bath;natatorium', 'stool', 'barrel;cask',
    'basket;handbasket', 'waterfall;falls', 'tent;collapsible;shelter', 'bag',
    'minibike;motorbike', 'cradle', 'oven', 'ball', 'food;solid;food',
    'step;stair', 'tank;storage;tank', 'trade;name;brand;name;brand;marque',
    'microwave;microwave;oven', 'pot;flowerpot',
    'animal;animate;being;beast;brute;creature;fauna',
    'bicycle;bike;wheel;cycle', 'lake',
    'dishwasher;dish;washer;dishwashing;machine',
    'screen;silver;screen;projection;screen', 'blanket;cover', 'sculpture',
    'hood;exhaust;hood', 'sconce', 'vase',
    'traffic;light;traffic;signal;stoplight', 'tray',
    'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin',
    'fan', 'pier;wharf;wharfage;dock', 'crt;screen', 'plate',
    'monitor;monitoring;device', 'bulletin;board;notice;board', 'shower',
    'radiator', 'glass;drinking;glass', 'clock', 'flag'
]

#  0 [                          wall] occ 46.14% (din 100%: 46.14%)
#   5 [                       ceiling] occ 16.45% (din 100%: 62.59%)
#   3 [                floor;flooring] occ 10.95% (din 100%: 73.55%)
#  14 [door;double;door+screen;door;screen] occ 5.00% (din 100%: 78.54%)
#   8 [             windowpane;window] occ 3.48% (din 100%: 82.02%)
#  10 [cabinet+wardrobe;closet;press+chest;of;drawers;chest;bureau;dresser] occ 3.51% (din 100%: 85.54%)
#  23 [             sofa;couch;lounge] occ 1.59% (din 100%: 87.13%)
#  38 [railing;rail+escalator;moving;staircase;moving;stairway+stairway;staircase+stairs;steps] occ 1.87% (din 100%: 89.00%)
#  15 [table+coffee;table;cocktail;table] occ 1.38% (din 100%: 90.38%)
#   7 [                           bed] occ 1.03% (din 100%: 91.41%)
#  22 [              painting;picture] occ 0.96% (din 100%: 92.37%)
#  19 [chair+armchair+swivel;chair+seat] occ 1.52% (din 100%: 93.88%)


def analyze_cls():
    all_classes_occurences_np = np.array(all_classes_occurences_ade).astype(
        int)
    all_occ = all_classes_occurences_np.sum(axis=0)[1]
    dict_all_cls = dict(all_classes_occurences_np)

    # combine
    combine_list = [(19, 30, 75, 31), (39, 57), (118, 124), (14, 58), (15, 64),
                    (17, 66, 125, 4, 72), (38, 96, 59, 53), (10, 35, 44)]
    for tuple in combine_list:
        first_elem = tuple[0]
        for elem in tuple[1:]:
            dict_all_cls[first_elem] += dict_all_cls[elem]
            dict_all_cls[elem] = 0
            LABELS_NAME[first_elem] += "+" + LABELS_NAME[elem]
            LABELS_NAME[elem] = ""

    total_occ = 0
    w_times = []
    for cls in dict_all_cls.keys():
        occ = dict_all_cls[cls]
        if occ == 0:
            continue
        rel_occ = occ * 100 / all_occ
        total_occ += rel_occ
        print("%3d [%30s] occ %.2f%% (din 100%%: %.2f%%) w_times=%.2f" %
              (cls, LABELS_NAME[int(cls)], rel_occ, total_occ, 100. / rel_occ))
        w_times.append(100. / rel_occ)
        if total_occ > 93:
            break
    print("w_times", w_times)


class SSegHRNet_v2(BasicExpert):
    def __init__(self, dataset_name, full_expert=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if full_expert:
            self.trained_num_class = 150
            # Network Builders
            self.encoder = ModelBuilder.build_encoder(
                arch="hrnetv2",
                fc_dim=720,
                weights="%s/hrnet_encoder_epoch_30.pth" % ss_model_path)
            self.decoder = ModelBuilder.build_decoder(
                arch="c1",
                fc_dim=720,
                num_class=self.trained_num_class,
                weights="%s/hrnet_decoder_epoch_30.pth" % ss_model_path,
                use_softmax=True)

            self.encoder.eval()
            self.decoder.eval()

            self.encoder.to(device)
            self.decoder.to(device)
        if dataset_name == "taskonomy":
            weights = [
                0.03933823, 0.11031396, 0.16570427, 0.3632622, 0.52115117,
                0.51646098, 1.14297737, 0.96902266, 1.31444027, 1.76809316,
                1.89247963, 1.1967561
            ]
            self.classes_weights = torch.tensor(weights).to(self.device)
        else:
            self.classes_weights = None

        if dataset_name == "hypersim":
            self.exp_annotations_2_common_annotations = torch.tensor(
                ADE20k_2_NYU19).long().cuda()
            self.gt_annotations_2_common_annotations = torch.tensor(
                NYU40_2_NYU19).long().cuda()
            self.n_obj_classes = 20
        else:
            assert (
                'semantic segmentation cls not implemented for this dataset')
            self.exp_annotations_2_common_annotations = None
            self.gt_annotations_2_common_annotations = None
            self.n_obj_classes = 10

        self.domain_name = "sem_seg"
        self.n_maps = 1
        self.n_final_maps = self.n_obj_classes

        self.str_id = "hrnet_v2"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        with torch.no_grad():
            batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
            sseg_maps = self.decoder(self.encoder(batch_rgb_frames.to(
                self.device),
                                                  return_feature_maps=True),
                                     segSize=256)

            class_labels = sseg_maps.argmax(dim=1, keepdim=True)
            class_labels = self.exp_annotations_2_common_annotations[
                class_labels] - 1

            class_labels = np.array(
                class_labels.data.cpu().numpy()).astype('long')
        return class_labels

    def get_task_type(self):
        return BasicExpert.TASK_CLASSIFICATION

    def no_maps_as_nn_input(self):
        return 1

    def no_maps_as_nn_output(self):
        return self.n_final_maps

    def no_maps_as_ens_input(self):
        return self.n_final_maps

    def gt_train_transform(self, gt_maps):
        # TODO check this
        return gt_maps.squeeze(1).long()

    def gt_eval_transform(self, inp_1chan_cls, n_classes):
        inp_1chan_cls = inp_1chan_cls.squeeze(1).long()
        bs, h, w = inp_1chan_cls.shape

        outp_multichan = torch.zeros(
            (bs, n_classes, h, w)).to(inp_1chan_cls.device).float()
        for chan in range(n_classes):
            outp_multichan[:, chan][inp_1chan_cls == chan] = 1.
        return outp_multichan
        # TODO check this
        #return gt_maps.squeeze(1).long()

    def exp_eval_transform(self, inp_1chan_cls, n_classes):
        inp_1chan_cls = inp_1chan_cls.squeeze(1).long()
        bs, h, w = inp_1chan_cls.shape

        outp_multichan = torch.zeros(
            (bs, n_classes, h, w)).to(inp_1chan_cls.device).float()
        for chan in range(n_classes):
            outp_multichan[:, chan][inp_1chan_cls == chan] = 1.
        return outp_multichan
        # TODO check this
        #return gt_maps.squeeze(1).long()

    def gt_to_inp_transform(self, inp_1chan_cls, n_classes):
        inp_1chan_cls = inp_1chan_cls.squeeze(1).long()
        bs, h, w = inp_1chan_cls.shape

        outp_multichan = torch.zeros(
            (bs, n_classes, h, w)).to(inp_1chan_cls.device).float()
        for chan in range(n_classes):
            outp_multichan[:, chan][inp_1chan_cls == chan] = 1.
        return outp_multichan

    def postprocess_eval(self, nn_outp):
        '''
        POST PROCESSING eval - posprocess operations for evaluation (e.g. scale/normalize)
        '''
        f = torch.nn.Softmax(dim=1)
        nn_outp = f(nn_outp)
        return nn_outp  #nn_outp.clamp(min=0, max=1)

    def postprocess_ensemble_eval(self, nn_outp):
        return nn_outp.clamp(min=0, max=1)


class SSegHRNet(BasicExpert):
    def __init__(self, dataset_name, full_expert=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if full_expert:
            self.trained_num_class = 150
            # Network Builders
            self.encoder = ModelBuilder.build_encoder(
                arch="hrnetv2",
                fc_dim=720,
                weights="%s/hrnet_encoder_epoch_30.pth" % ss_model_path)
            self.decoder = ModelBuilder.build_decoder(
                arch="c1",
                fc_dim=720,
                num_class=self.trained_num_class,
                weights="%s/hrnet_decoder_epoch_30.pth" % ss_model_path,
                use_softmax=True)

            self.encoder.eval()
            self.decoder.eval()

            self.encoder.to(device)
            self.decoder.to(device)

        if dataset_name == "taskonomy":
            weights = [
                0.03933823, 0.11031396, 0.16570427, 0.3632622, 0.52115117,
                0.51646098, 1.14297737, 0.96902266, 1.31444027, 1.76809316,
                1.89247963, 1.1967561
            ]
            self.classes_weights = torch.tensor(weights).to(self.device)
        else:
            self.classes_weights = None

        self.combine_list = [(19, 30, 75, 31), (14, 58), (15, 64),
                             (38, 96, 59, 53), (10, 35, 44)]
        self.to_keep_list = [0, 5, 3, 14, 8, 10, 23, 38, 15, 7, 22, 19]
        # self.to_keep_list = [0, 5, 3, 14, 8, 10, 23, 38, 15, 7, 22, 19]
        #  0 [                          wall] occ 46.14% (din 100%: 46.14%)
        #   5 [                       ceiling] occ 16.45% (din 100%: 62.59%)
        #   3 [                floor;flooring] occ 10.95% (din 100%: 73.55%)
        #  14 [door;double;door+screen;door;screen] occ 5.00% (din 100%: 78.54%)
        #   8 [             windowpane;window] occ 3.48% (din 100%: 82.02%)
        #  10 [cabinet+wardrobe;closet;press+chest;of;drawers;chest;bureau;dresser] occ 3.51% (din 100%: 85.54%)
        #  23 [             sofa;couch;lounge] occ 1.59% (din 100%: 87.13%)
        #  38 [railing;rail+escalator;moving;staircase;moving;stairway+stairway;staircase+stairs;steps] occ 1.87% (din 100%: 89.00%)
        #  15 [table+coffee;table;cocktail;table] occ 1.38% (din 100%: 90.38%)
        #   7 [                           bed] occ 1.03% (din 100%: 91.41%)
        #  22 [              painting;picture] occ 0.96% (din 100%: 92.37%)
        #  19 [chair+armchair+swivel;chair+seat] occ 1.52% (din 100%: 93.88%)
        self.domain_name = "sem_seg"
        self.n_maps = 1
        self.n_final_maps = len(self.to_keep_list)

        self.str_id = "hrnet"
        self.identifier = self.domain_name + "_" + self.str_id

    def normalize_output_fcn(self, outp):
        return outp

    def get_task_type(self):
        return BasicExpert.TASK_CLASSIFICATION

    def apply_expert_batch(self, batch_rgb_frames):
        with torch.no_grad():
            # analyze_cls()
            batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
            sseg_maps = self.decoder(self.encoder(batch_rgb_frames.to(
                self.device),
                                                  return_feature_maps=True),
                                     segSize=256)

            # combine outputs
            for tuple in self.combine_list:
                first_elem = tuple[0]
                for elem in tuple[1:]:
                    sseg_maps[:, first_elem] += sseg_maps[:, elem]
                    sseg_maps[:, elem] = 0

            class_labels = sseg_maps[:, self.to_keep_list].argmax(dim=1)[:,
                                                                         None]

            class_labels = np.array(
                class_labels.data.cpu().numpy()).astype('long')
        return class_labels

    def no_maps_as_nn_input(self):
        return 1

    def no_maps_as_nn_output(self):
        return len(self.to_keep_list)

    def no_maps_as_ens_input(self):
        return len(self.to_keep_list)

    def gt_train_transform(self, gt_maps):
        return gt_maps.squeeze(1).long()

    def gt_eval_transform(self, gt_maps, n_classes):
        return gt_maps.squeeze(1).long()

    def exp_eval_transform(self, gt_maps, n_classes):
        return gt_maps.squeeze(1).long()

    def gt_to_inp_transform(self, inp_1chan_cls, n_classes):
        inp_1chan_cls = inp_1chan_cls.squeeze(1).long()
        bs, h, w = inp_1chan_cls.shape

        outp_multichan = torch.zeros(
            (bs, n_classes, h, w)).to(inp_1chan_cls.device).float()
        for chan in range(n_classes):
            outp_multichan[:, chan][inp_1chan_cls == chan] = 1.
        return outp_multichan
