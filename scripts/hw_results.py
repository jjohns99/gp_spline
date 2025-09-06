import numpy as np
import matplotlib.pyplot as plt

def main():
  save_figs=True

  run = 3

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsfonts, siunitx}'
  })

  # [pos_rmse, rot_rmse, solve_time, iters]
  se3spl_4_01_none = np.array([[0.0354615, 0.0342801, 0.0131211, 5],
                               [0.0567211, 0.0474366, 0.0114632, 5],
                               [0.0433965, 0.0429128, 0.011816, 5],
                               [0.0363502, 0.043148, 0.011986, 5],
                               [0.0449694, 0.0447, 0.0142438, 7],
                               [0.0630577, 0.0385946, 0.01402, 6],
                               [0.0482748, 0.0358637, 0.0117519, 5],
                               [0.070835, 0.0602564, 0.0222571, 13],
                               [0.0702812, 0.0551063, 0.0107028, 5],
                               [0.056544, 0.0368269, 0.0380142, 20],
                               [0.0467335, 0.0310682, 0.013243, 6]])
  se3spl_4_01_mp = np.array([[0.0287955, 0.0268013, 0.0410109, 6],
                             [0.0426409, 0.0351649, 0.034514, 5],
                             [0.0300579, 0.034564, 0.0400198, 6],
                             [0.0251896, 0.0358871, 0.0353451, 5],
                             [0.0341679, 0.0382885, 0.0319722, 4],
                             [0.0591681, 0.0364354, 0.042002, 6],
                             [0.0334884, 0.0263785, 0.0352139, 5],
                             [0.0654863, 0.0551311, 0.0441508, 5],
                             [0.0479952, 0.0416414, 0.037508, 6],
                             [0.0272042, 0.0269049, 0.0699501, 7],
                             [0.0310907, 0.0242766, 0.0405998, 6]])
  se3spl_4_01_imu = np.array([[0.0138089, 0.0173181, 0.298997, 12],
                              [0.0250518, 0.0211659, 0.244034, 9],
                              [0.0162778, 0.0266663, 0.19333, 7],
                              [0.0168496, 0.0298682, 0.469984, 22],
                              [0.0190497, 0.0330582, 0.169759, 6],
                              [0.0323216, 0.020413, 0.329281, 13],
                              [0.0210959, 0.0165306, 0.281017, 12],
                              [0.016446, 0.0244467, 0.16955, 6],
                              [0.0255356, 0.0299765, 0.259216, 10],
                              [0.0178453, 0.0215353, 0.454289, 26],
                              [0.0183616, 0.0158483, 0.456859, 23]])
  se3spl_4_01_imump = np.array([[0.0135992, 0.0168009, 0.240698, 8],
                                [0.0247858, 0.0210249, 0.265513, 9],
                                [0.0162133, 0.0266228, 0.209764, 7],
                                [0.0170152, 0.0297876, 0.33318, 14],
                                [0.0190268, 0.032967, 0.15564, 5],
                                [0.0325926, 0.0205273, 0.211835, 7],
                                [0.021422, 0.0167648, 0.209036, 7],
                                [0.0163701, 0.0241425, 0.183331, 6],
                                [0.0257005, 0.0299097, 0.228683, 8],
                                [0.0176683, 0.0215327, 0.262523, 8],
                                [0.0187221, 0.0161683, 0.182411, 6]])

  se3spl_6_01_none = np.array([[0.0323385, 0.0297691, 0.0879641, 34],
                               [0.0556307, 0.0459641, 0.022548, 6],
                               [0.0412764, 0.0410707, 0.0325401, 11],
                               [0.0310448, 0.0387375, 0.0310001, 8],
                               [0.044188, 0.0439126, 0.030504, 9],
                               [0.0610474, 0.0385336, 0.0343852, 9],
                               [0.0447221, 0.0349264, 0.0745702, 23],
                               [0.0701648, 0.0578922, 0.0237868, 7],
                               [0.0697468, 0.0582777, 0.0745749, 24],
                               [0.0431579, 0.0342025, 0.0884612, 21],
                               [0.0423125, 0.0315105, 0.0348971, 9]])
  se3spl_6_01_mp = np.array([[0.0288462, 0.0268085, 0.243466, 21],
                             [0.0428336, 0.0355724, 0.80775, 19],
                             [0.0302498, 0.0347436, 0.251826, 6],
                             [0.0252089, 0.0358509, 0.090199, 5],
                             [0.0341952, 0.0382559, 0.099112, 6],
                             [0.0590576, 0.036357, 0.497574, 38],
                             [0.0333076, 0.0263018, 0.289702, 15],
                             [0.0656402, 0.055258, 0.900781, 28],
                             [0.0480642, 0.0416869, 0.268962, 34],
                             [0.0272975, 0.0270219, 0.587319, 9],
                             [0.0311247, 0.0244287, 0.372245, 8]])
  se3spl_6_01_imu = np.array([[0.0133385, 0.0167893, 0.953843, 28],
                              [0.0251095, 0.0211787, 1.3552, 51],
                              [0.0162549, 0.0267653, 0.466029, 17],
                              [0.0170776, 0.0299276, 0.663814, 20],
                              [0.0188712, 0.0329814, 0.651793, 25],
                              [0.032378, 0.0204852, 0.80673, 22],
                              [0.0210803, 0.0165884, 0.670673, 20],
                              [0.0163753, 0.0244471, 0.640214, 18],
                              [0.0255529, 0.0300558, 0.682353, 25],
                              [0.0177701, 0.0215947, 0.727942, 19],
                              [0.0184206, 0.0160019, 0.578824, 15]])
  se3spl_6_01_imump = np.array([[0.0132932, 0.0165743, 4.98017, 41],
                                [0.0247703, 0.0209505, 1.52665, 13],
                                [0.0162627, 0.0266158, 1.15748, 16],
                                [0.0170008, 0.0297798, 1.10689, 10],
                                [0.01902, 0.0329486, 1.11444, 8],
                                [0.0325744, 0.0204857, 1.89089, 22],
                                [0.0214167, 0.0168019, 2.36127, 20],
                                [0.0163686, 0.024132, 1.8915, 13],
                                [0.0257392, 0.0299388, 2.37696, 22],
                                [0.0176248, 0.0214761, 2.23472, 17],
                                [0.0186274, 0.0161549, 2.62825, 23]])

  so3xr3spl_4_01_none = np.array([[0.034998, 0.0338756, 0.016072, 9],
                                  [0.0570362, 0.0473875, 0.017159, 10],
                                  [0.0425401, 0.0424201, 0.0185721, 13],
                                  [0.0360428, 0.0428114, 0.020303, 11],
                                  [0.0454426, 0.04428, 0.0105822, 7],
                                  [0.0664139, 0.038594, 0.0178549, 10],
                                  [0.0487659, 0.0359569, 0.0161521, 9],
                                  [0.0708901, 0.0603066, 0.017571, 13],
                                  [0.0718931, 0.0546839, 0.0207841, 17],
                                  [0.0553294, 0.0359592, 0.025166, 19],
                                  [0.0527903, 0.030843, 0.0167258, 10]])
  so3xr3spl_4_01_mp = np.array([[0.0288754, 0.0269068, 0.0288341, 4],
                                [0.0427648, 0.0351906, 0.0288119, 4],
                                [0.0299791, 0.0345782, 0.0303121, 4],
                                [0.0251624, 0.0357811, 0.0320842, 4],
                                [0.0340909, 0.0382736, 0.0213208, 4],
                                [0.0585413, 0.0360149, 0.0296338, 4],
                                [0.0334984, 0.026405, 0.0279453, 4],
                                [0.0656519, 0.0552569, 0.0280428, 4],
                                [0.0480829, 0.0416928, 0.0343828, 5],
                                [0.0271484, 0.0269656, 0.0363679, 4],
                                [0.0308441, 0.0241792, 0.0280259, 4]])
  so3xr3spl_4_01_imu = np.array([[0.013712, 0.0171144, 0.112264, 10],
                                [0.0249532, 0.0210075, 0.0761039, 6],
                                [0.0162973, 0.0267529, 0.0679948, 5],
                                [0.0169151, 0.0298961, 0.104028, 9],
                                [0.0194817, 0.0332205, 0.0683091, 6],
                                [0.0323899, 0.020392, 0.0674713, 5],
                                [0.0211674, 0.0164817, 0.0753422, 6],
                                [0.0166058, 0.0244678, 0.0723889, 6],
                                [0.0255739, 0.0301532, 0.0766089, 6],
                                [0.0176849, 0.0214255, 0.115229, 15],
                                [0.0184757, 0.0159656, 0.0971501, 14]])
  so3xr3spl_4_01_imump = np.array([[0.0137001, 0.0168863, 0.101328, 8],
                                   [0.024732, 0.0209712, 0.084048, 6],
                                   [0.016209, 0.026637, 0.0745127, 5],
                                   [0.0170109, 0.0297965, 0.114996, 9],
                                   [0.0191347, 0.0330011, 0.0725608, 5],
                                   [0.0325962, 0.0205122, 0.0846491, 5],
                                   [0.0214827, 0.0167701, 0.0818419, 5],
                                   [0.0163976, 0.0241312, 0.070529, 5],
                                   [0.0257158, 0.0299414, 0.106184, 5],
                                   [0.0175952, 0.0215099, 0.0978673, 5],
                                   [0.018756, 0.0161875, 0.0636919, 4]])

  so3xr3spl_6_01_none = np.array([[0.0323758, 0.0300326, 0.082732, 26],
                                  [0.0556515, 0.0458843, 0.0192571, 7],
                                  [0.0410974, 0.0410061, 0.0231509, 8],
                                  [0.0312688, 0.0386804, 0.02371, 8],
                                  [0.0439909, 0.0436112, 0.0312731, 17],
                                  [0.0584008, 0.0355383, 0.0617301, 32],
                                  [0.0531768, 0.041764, 0.063236, 31],
                                  [0.0701364, 0.0578538, 0.0217359, 9],
                                  [0.069811, 0.057898, 0.0355599, 19],
                                  [0.0970034, 0.0585467, 0.062254, 32],
                                  [0.0437685, 0.0310179, 0.025475, 10]])
  so3xr3spl_6_01_mp = np.array([[0.0289066, 0.0268979, 0.0870588, 4],
                                [0.0429539, 0.0355913, 0.242193, 9],
                                [0.0301718, 0.0347609, 0.134242, 5],
                                [0.0251751, 0.0357547, 0.520493, 51],
                                [0.0340867, 0.0382139, 0.0654452, 5],
                                [0.0583142, 0.0358679, 0.230132, 5],
                                [0.0332993, 0.0263227, 0.377428, 7],
                                [0.0658324, 0.0554106, 0.224002, 5],
                                [0.0481011, 0.0416606, 0.163366, 22],
                                [0.0271607, 0.0270562, 0.204312, 5],
                                [0.0308416, 0.0242903, 0.156415, 5]])
  so3xr3spl_6_01_imu = np.array([[0.0139187, 0.0173788, 0.192545, 8],
                                [0.024935, 0.0210973, 0.365605, 37],
                                [0.0162894, 0.0266942, 0.272105, 22],
                                [0.0169134, 0.0299028, 0.17028, 14],
                                [0.0194505, 0.0331887, 0.309451, 28],
                                [0.032376, 0.0204052, 0.290568, 19],
                                [0.0210862, 0.0165226, 0.332597, 27],
                                [0.0165662, 0.0244201, 0.20667, 20],
                                [0.0255775, 0.0301398, 0.344434, 27],
                                [0.0175903, 0.0214159, 0.456818, 33],
                                [0.0184646, 0.0159848, 0.27272, 22]])
  so3xr3spl_6_01_imump = np.array([[0.0135003, 0.016747, 0.539104, 7],
                                  [0.0247016, 0.0208774, 0.227119, 5],
                                  [0.0162735, 0.0266216, 0.386214, 4],
                                  [0.0170192, 0.0298013, 0.471707, 6],
                                  [0.0192433, 0.0330145, 0.300255, 5],
                                  [0.0325405, 0.0204626, 0.388136, 6],
                                  [0.0214701, 0.01682, 0.332853, 5],
                                  [0.0163837, 0.0241153, 0.291825, 5],
                                  [0.0257983, 0.0299808, 0.247921, 4],
                                  [0.0175446, 0.0214234, 0.501944, 6],
                                  [0.0186418, 0.0161581, 0.594789, 5]])

  se3gp_none = np.array([[0.0309463, 0.02898, 0.00325799, 5],
                        [0.0518501, 0.0436354, 0.00296092, 5],
                        [0.0392907, 0.0404291, 0.00317192, 5],
                        [0.0283493, 0.038033, 0.00269485, 4],
                        [0.0414124, 0.042722, 0.00289202, 6],
                        [0.0573138, 0.0370568, 0.00314212, 5],
                        [0.0416893, 0.0340205, 0.00319195, 5],
                        [0.0674217, 0.0564299, 0.00248194, 5],
                        [0.0650688, 0.0504313, 0.00272512, 5],
                        [0.0388016, 0.0332957, 0.00291014, 5],
                        [0.0384012, 0.0302359, 0.00285697, 5]])
  se3gp_mp = np.array([[0.0289191, 0.0268846, 0.0220921, 5],
                      [0.0426838, 0.0352253, 0.0156691, 4],
                      [0.0301018, 0.0345189, 0.0223181, 6],
                      [0.0252018, 0.0358278, 0.0157452, 4],
                      [0.0342921, 0.03824, 0.021224, 6],
                      [0.05937, 0.0365223, 0.02209, 6],
                      [0.0336023, 0.0264909, 0.0195632, 5],
                      [0.0654994, 0.0551341, 0.0206792, 6],
                      [0.0480002, 0.0416461, 0.0182288, 5],
                      [0.0272009, 0.0269637, 0.01858, 5],
                      [0.0310798, 0.0242319, 0.0184639, 5]])
  se3gp_imu = np.array([[0.0142621, 0.017113, 0.12291, 5],
                        [0.0252632, 0.0207828, 0.123703, 4],
                        [0.0164298, 0.0265907, 0.125597, 5],
                        [0.0170651, 0.0293718, 0.0935183, 4],
                        [0.0195986, 0.0330191, 0.12492, 5],
                        [0.0326363, 0.0198685, 0.122792, 5],
                        [0.0212016, 0.0160719, 0.0997362, 4],
                        [0.0165493, 0.0238451, 0.116121, 5],
                        [0.0257348, 0.0296791, 0.118258, 5],
                        [0.0176863, 0.021023, 0.0996656, 4],
                        [0.0183746, 0.0154552, 0.09902, 4]])
  se3gp_imump = np.array([[0.0139458, 0.0171007, 0.151158, 5],
                          [0.0247785, 0.0211362, 0.117408, 4],
                          [0.0162625, 0.0266258, 0.129592, 5],
                          [0.0170214, 0.0297543, 0.09975, 4],
                          [0.0192659, 0.0330361, 0.129904, 5],
                          [0.032591, 0.020504, 0.130516, 5],
                          [0.0215429, 0.0168106, 0.106042, 4],
                          [0.0161671, 0.0240456, 0.123551, 5],
                          [0.0257342, 0.0299126, 0.125539, 5],
                          [0.0175898, 0.0214322, 0.130264, 5],
                          [0.0187401, 0.0162357, 0.13043, 5]])

  so3xr3gp_none = np.array([[0.0309493, 0.0289832, 0.00417495, 6],
                            [0.0518402, 0.0436483, 0.00360203, 6],
                            [0.0392989, 0.0404365, 0.00383806, 6],
                            [0.0283603, 0.038046, 0.00334001, 5],
                            [0.0414091, 0.0427143, 0.00290799, 6],
                            [0.0573222, 0.0370669, 0.00378108, 6],
                            [0.041682, 0.034016, 0.00376797, 6],
                            [0.0674486, 0.056442, 0.00288081, 6],
                            [0.0650294, 0.0504336, 0.00315499, 6],
                            [0.038761, 0.0332781, 0.00344205, 6],
                            [0.0384134, 0.0302453, 0.00341105, 6]])
  so3xr3gp_mp = np.array([[0.0290128, 0.0270068, 0.016885, 5],
                          [0.0428044, 0.0352394, 0.0146239, 4],
                          [0.0300744, 0.0345841, 0.0193031, 6],
                          [0.0251506, 0.0357436, 0.0123761, 4],
                          [0.0342037, 0.0382208, 0.0152411, 6],
                          [0.0585775, 0.0360326, 0.0147438, 5],
                          [0.0336247, 0.0265427, 0.014725, 5],
                          [0.0656963, 0.0552745, 0.0150061, 6],
                          [0.0480889, 0.0416921, 0.0135951, 5],
                          [0.0271797, 0.0270318, 0.0141811, 5],
                          [0.0308738, 0.0241612, 0.014271, 5]])
  so3xr3gp_imu = np.array([[0.0141274, 0.0169812, 0.094902, 5],
                          [0.0252549, 0.0207908, 0.0853369, 5],
                          [0.0164232, 0.0265999, 0.0853479, 5],
                          [0.0170559, 0.0293481, 0.066946, 4],
                          [0.0196178, 0.0330679, 0.0839777, 5],
                          [0.0326187, 0.0198481, 0.0852282, 5],
                          [0.0212028, 0.0160368, 0.071305, 4],
                          [0.0165384, 0.0238323, 0.080004, 5],
                          [0.0256936, 0.0296732, 0.081548, 5],
                          [0.0176715, 0.0209861, 0.0700588, 4],
                          [0.0183692, 0.0154426, 0.0698211, 4]])
  so3xr3gp_imump = np.array([[0.0139628, 0.0171172, 0.102093, 5],
                            [0.0247567, 0.0210836, 0.0893588, 5],
                            [0.0162665, 0.0266289, 0.089252, 5],
                            [0.0170862, 0.0297701, 0.0839262, 5],
                            [0.0192692, 0.0330417, 0.087873, 5],
                            [0.0325935, 0.0205055, 0.0891171, 5],
                            [0.0215694, 0.016829, 0.0737839, 4],
                            [0.0161692, 0.0240378, 0.0834332, 5],
                            [0.0257524, 0.0299191, 0.085355, 5],
                            [0.0175706, 0.0214254, 0.0883718, 5],
                            [0.0187327, 0.0162318, 0.088408, 5]])

  # none_pos_rmse = [se3spl_4_01_none.item(run,0), se3spl_6_01_none.item(run,0), se3gp_none.item(run,0), so3xr3spl_4_01_none.item(run,0), so3xr3spl_6_01_none.item(run,0), so3xr3gp_none.item(run,0)]
  # mp_pos_rmse = [se3spl_4_01_mp.item(run,0), se3spl_6_01_mp.item(run,0), se3gp_mp.item(run,0), so3xr3spl_4_01_mp.item(run,0), so3xr3spl_6_01_mp.item(run,0), so3xr3gp_mp.item(run,0)]
  # imu_pos_rmse = [se3spl_4_01_imu.item(run,0), se3spl_6_01_imu.item(run,0), se3gp_imu.item(run,0), so3xr3spl_4_01_imu.item(run,0), so3xr3spl_6_01_imu.item(run,0), so3xr3gp_imu.item(run,0)]
  # imump_pos_rmse = [se3spl_4_01_imump.item(run,0), se3spl_6_01_imump.item(run,0), se3gp_imump.item(run,0), so3xr3spl_4_01_imump.item(run,0), so3xr3spl_6_01_imump.item(run,0), so3xr3gp_imump.item(run,0)]

  # none_rot_rmse = [se3spl_4_01_none.item(run,1), se3spl_6_01_none.item(run,1), se3gp_none.item(run,1), so3xr3spl_4_01_none.item(run,1), so3xr3spl_6_01_none.item(run,1), so3xr3gp_none.item(run,1)]
  # mp_rot_rmse = [se3spl_4_01_mp.item(run,1), se3spl_6_01_mp.item(run,1), se3gp_mp.item(run,1), so3xr3spl_4_01_mp.item(run,1), so3xr3spl_6_01_mp.item(run,1), so3xr3gp_mp.item(run,1)]
  # imu_rot_rmse = [se3spl_4_01_imu.item(run,1), se3spl_6_01_imu.item(run,1), se3gp_imu.item(run,1), so3xr3spl_4_01_imu.item(run,1), so3xr3spl_6_01_imu.item(run,1), so3xr3gp_imu.item(run,1)]
  # imump_rot_rmse = [se3spl_4_01_imump.item(run,1), se3spl_6_01_imump.item(run,1), se3gp_imump.item(run,1), so3xr3spl_4_01_imump.item(run,1), so3xr3spl_6_01_imump.item(run,1), so3xr3gp_imump.item(run,1)]

  # none_time = [se3spl_4_01_none.item(run,2), se3spl_6_01_none.item(run,2), se3gp_none.item(run,2), so3xr3spl_4_01_none.item(run,2), so3xr3spl_6_01_none.item(run,2), so3xr3gp_none.item(run,2)]
  # mp_time = [se3spl_4_01_mp.item(run,2), se3spl_6_01_mp.item(run,2), se3gp_mp.item(run,2), so3xr3spl_4_01_mp.item(run,2), so3xr3spl_6_01_mp.item(run,2), so3xr3gp_mp.item(run,2)]
  # imu_time = [se3spl_4_01_imu.item(run,2), se3spl_6_01_imu.item(run,2), se3gp_imu.item(run,2), so3xr3spl_4_01_imu.item(run,2), so3xr3spl_6_01_imu.item(run,2), so3xr3gp_imu.item(run,2)]
  # imump_time = [se3spl_4_01_imump.item(run,2), se3spl_6_01_imump.item(run,2), se3gp_imump.item(run,2), so3xr3spl_4_01_imump.item(run,2), so3xr3spl_6_01_imump.item(run,2), so3xr3gp_imump.item(run,2)]

  # none_iters = [se3spl_4_01_none.item(run,3), se3spl_6_01_none.item(run,3), se3gp_none.item(run,3), so3xr3spl_4_01_none.item(run,3), so3xr3spl_6_01_none.item(run,3), so3xr3gp_none.item(run,3)]
  # mp_iters = [se3spl_4_01_mp.item(run,3), se3spl_6_01_mp.item(run,3), se3gp_mp.item(run,3), so3xr3spl_4_01_mp.item(run,3), so3xr3spl_6_01_mp.item(run,3), so3xr3gp_mp.item(run,3)]
  # imu_iters = [se3spl_4_01_imu.item(run,3), se3spl_6_01_imu.item(run,3), se3gp_imu.item(run,3), so3xr3spl_4_01_imu.item(run,3), so3xr3spl_6_01_imu.item(run,3), so3xr3gp_imu.item(run,3)]
  # imump_iters = [se3spl_4_01_imump.item(run,3), se3spl_6_01_imump.item(run,3), se3gp_imump.item(run,3), so3xr3spl_4_01_imump.item(run,3), so3xr3spl_6_01_imump.item(run,3), so3xr3gp_imump.item(run,3)]


  # labels = [r'SE(3) spline, $k$ = 4', r'SE(3) spline, $k$ = 6', r'SE(3) GP', r'SO(3)$\times \mathbb{R}^3$ spline, $k$ = 4', r'SO(3)$\times \mathbb{R}^3$ spline, $k$ = 6', r'SO(3)$\times \mathbb{R}^3$ GP']
  # appended_labels = labels + ['',''] + labels + ['',''] + labels + ['',''] + labels
  # colors = ['red', 'green', 'blue', 'lightcoral', 'lightgreen', 'lightblue']
  # appended_colors = colors + ['k','k'] + colors + ['k','k'] + colors + ['k','k'] + colors

  # fig, ax = plt.subplots(4,1, figsize=(12,7))
  # ax[0].bar(range(30), none_pos_rmse + [0.0, 0.0] + mp_pos_rmse + [0.0, 0.0] + imu_pos_rmse + [0.0, 0.0] + imump_pos_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  # ax[1].bar(range(30), none_rot_rmse + [0.0, 0.0] + mp_rot_rmse + [0.0, 0.0] + imu_rot_rmse + [0.0, 0.0] + imump_rot_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  # ax[2].bar(range(30), none_time + [0.0, 0.0] + mp_time + [0.0, 0.0] + imu_time + [0.0, 0.0] + imump_time, 1.0, color=appended_colors, edgecolor='k')
  # ax[3].bar(range(30), none_iters + [0.0, 0.0] + mp_iters + [0.0, 0.0] + imu_iters + [0.0, 0.0] + imump_iters, 1.0, color=appended_colors, edgecolor='k')
  # ax[0].set_xticks([])
  # ax[1].set_xticks([])
  # ax[2].set_xticks([])
  # ax[3].set_xticks([2.5, 10.5, 18.5, 26.5])
  # ax[3].set_xticklabels([r'None', r'Motion Priors', r'IMU', r'Motion Priors + IMU'])
  # ax[0].set_ylabel(r'Pos. RMSE (\si{m})')
  # ax[1].set_ylabel(r'Rot. RMSE (\si{rad})')
  # ax[2].set_ylabel(r'Solve Time (\si{s})')
  # ax[3].set_ylabel(r'Iterations')
  # ax[3].set_xlabel(r'Regularization Type', fontsize=12)

  # legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=c, edgecolor='k') for c in colors]
  # ax[0].legend(legend_handles, labels, ncol=2)
  # ax[0].set_title(r'Hardware Trajectory')

  none_pos_rmse = [se3spl_4_01_none[:,0], se3spl_6_01_none[:,0], se3gp_none[:,0], so3xr3spl_4_01_none[:,0], so3xr3spl_6_01_none[:,0], so3xr3gp_none[:,0]]
  mp_pos_rmse = [se3spl_4_01_mp[:,0], se3spl_6_01_mp[:,0], se3gp_mp[:,0], so3xr3spl_4_01_mp[:,0], so3xr3spl_6_01_mp[:,0], so3xr3gp_mp[:,0]]
  imu_pos_rmse = [se3spl_4_01_imu[:,0], se3spl_6_01_imu[:,0], se3gp_imu[:,0], so3xr3spl_4_01_imu[:,0], so3xr3spl_6_01_imu[:,0], so3xr3gp_imu[:,0]]
  imump_pos_rmse = [se3spl_4_01_imump[:,0], se3spl_6_01_imump[:,0], se3gp_imump[:,0], so3xr3spl_4_01_imump[:,0], so3xr3spl_6_01_imump[:,0], so3xr3gp_imump[:,0]]

  none_rot_rmse = [se3spl_4_01_none[:,1], se3spl_6_01_none[:,1], se3gp_none[:,1], so3xr3spl_4_01_none[:,1], so3xr3spl_6_01_none[:,1], so3xr3gp_none[:,1]]
  mp_rot_rmse = [se3spl_4_01_mp[:,1], se3spl_6_01_mp[:,1], se3gp_mp[:,1], so3xr3spl_4_01_mp[:,1], so3xr3spl_6_01_mp[:,1], so3xr3gp_mp[:,1]]
  imu_rot_rmse = [se3spl_4_01_imu[:,1], se3spl_6_01_imu[:,1], se3gp_imu[:,1], so3xr3spl_4_01_imu[:,1], so3xr3spl_6_01_imu[:,1], so3xr3gp_imu[:,1]]
  imump_rot_rmse = [se3spl_4_01_imump[:,1], se3spl_6_01_imump[:,1], se3gp_imump[:,1], so3xr3spl_4_01_imump[:,1], so3xr3spl_6_01_imump[:,1], so3xr3gp_imump[:,1]]

  none_time = [se3spl_4_01_none[:,2], se3spl_6_01_none[:,2], se3gp_none[:,2], so3xr3spl_4_01_none[:,2], so3xr3spl_6_01_none[:,2], so3xr3gp_none[:,2]]
  mp_time = [se3spl_4_01_mp[:,2], se3spl_6_01_mp[:,2], se3gp_mp[:,2], so3xr3spl_4_01_mp[:,2], so3xr3spl_6_01_mp[:,2], so3xr3gp_mp[:,2]]
  imu_time = [se3spl_4_01_imu[:,2], se3spl_6_01_imu[:,2], se3gp_imu[:,2], so3xr3spl_4_01_imu[:,2], so3xr3spl_6_01_imu[:,2], so3xr3gp_imu[:,2]]
  imump_time = [se3spl_4_01_imump[:,2], se3spl_6_01_imump[:,2], se3gp_imump[:,2], so3xr3spl_4_01_imump[:,2], so3xr3spl_6_01_imump[:,2], so3xr3gp_imump[:,2]]

  none_iters = [se3spl_4_01_none[:,3], se3spl_6_01_none[:,3], se3gp_none[:,3], so3xr3spl_4_01_none[:,3], so3xr3spl_6_01_none[:,3], so3xr3gp_none[:,3]]
  mp_iters = [se3spl_4_01_mp[:,3], se3spl_6_01_mp[:,3], se3gp_mp[:,3], so3xr3spl_4_01_mp[:,3], so3xr3spl_6_01_mp[:,3], so3xr3gp_mp[:,3]]
  imu_iters = [se3spl_4_01_imu[:,3], se3spl_6_01_imu[:,3], se3gp_imu[:,3], so3xr3spl_4_01_imu[:,3], so3xr3spl_6_01_imu[:,3], so3xr3gp_imu[:,3]]
  imump_iters = [se3spl_4_01_imump[:,3], se3spl_6_01_imump[:,3], se3gp_imump[:,3], so3xr3spl_4_01_imump[:,3], so3xr3spl_6_01_imump[:,3], so3xr3gp_imump[:,3]]


  labels = [r'SE(3) spline, $k$ = 4', r'SE(3) spline, $k$ = 6', r'SE(3) GP', r'SO(3)$\times \mathbb{R}^3$ spline, $k$ = 4', r'SO(3)$\times \mathbb{R}^3$ spline, $k$ = 6', r'SO(3)$\times \mathbb{R}^3$ GP']
  appended_labels = labels + ['',''] + labels + ['',''] + labels + ['',''] + labels
  colors = ['red', 'green', 'blue', 'lightcoral', 'lightgreen', 'lightblue']
  appended_colors = colors + colors + colors + colors
  positions = [0,1,2,3,4,5,8,9,10,11,12,13,16,17,18,19,20,21,24,25,26,27,28,29]
  boxplots = []

  fig, ax = plt.subplots(4,1, figsize=(12,7))
  boxplots.append(ax[0].boxplot(none_pos_rmse + mp_pos_rmse + imu_pos_rmse + imump_pos_rmse, positions=positions, vert=True, patch_artist=True, sym="x", medianprops={'color': 'k'}))
  boxplots.append(ax[1].boxplot(none_rot_rmse + mp_rot_rmse + imu_rot_rmse + imump_rot_rmse, positions=positions, vert=True, patch_artist=True, sym="x", medianprops={'color': 'k'}))
  boxplots.append(ax[2].boxplot(none_time + mp_time + imu_time + imump_time, positions=positions, vert=True, patch_artist=True, sym="x", medianprops={'color': 'k'}))
  boxplots.append(ax[3].boxplot(none_iters + mp_iters + imu_iters + imump_iters, positions=positions, vert=True, patch_artist=True, sym="x", medianprops={'color': 'k'}))
  for boxplot in boxplots:
    for patch, color in zip(boxplot['boxes'], appended_colors):
      patch.set_facecolor(color)
      # patch.set_edgecolor(color)
    for flier, color in zip(boxplot['fliers'], appended_colors):
      flier.set_markeredgecolor(color)
    # for whisker, color in zip(boxplot['whiskers'], [c for c in appended_colors for _ in (0,1)]):
    #   whisker.set_color(color)
    # for cap, color in zip(boxplot['caps'], [c for c in appended_colors for _ in (0,1)]):
    #   cap.set_color(color)
    
  ax[0].set_yscale('log')
  ax[1].set_yscale('log')
  ax[2].set_yscale('log')
  ax[0].set_xticks([])
  ax[1].set_xticks([])
  ax[2].set_xticks([])
  ax[3].set_xticks([2.5, 10.5, 18.5, 26.5])
  ax[3].set_xticklabels([r'None', r'Motion Priors', r'IMU', r'Motion Priors + IMU'])
  ax[0].set_ylabel(r'Pos. RMSE (\si{m})')
  ax[1].set_ylabel(r'Rot. RMSE (\si{rad})')
  ax[2].set_ylabel(r'Solve Time (\si{s})')
  ax[3].set_ylabel(r'Iterations')
  ax[3].set_xlabel(r'Regularization Type', fontsize=12)
  # ax[2].set_ylim([-0.2, 2.8])

  legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=c, edgecolor='k') for c in colors]
  ax[0].legend(legend_handles, labels, ncol=2, bbox_to_anchor=(0.59, 1.15))
  ax[0].set_title(r'Hardware Trajectories')
  ax[0].grid(which='both', axis='y')
  ax[1].grid(which='both', axis='y')
  ax[2].grid(axis='y')
  ax[3].grid(axis='y')
  ax[3].set_yticks([0, 10, 20, 30, 40, 50])
  ax[3].set_ylim([0, 53])

  if save_figs:
    plt.savefig('hw_mc.png', dpi=300, bbox_inches="tight")

  plt.show()

if __name__ == "__main__":
  main()