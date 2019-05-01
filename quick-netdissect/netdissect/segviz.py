import numpy, scipy
import numpy as np
def segment_visualization(seg, size):

    result = numpy.zeros((seg.shape[1] * seg.shape[2], 3), dtype=numpy.uint8)
    flatseg = seg.reshape(seg.shape[0], seg.shape[1] * seg.shape[2])
    bc = numpy.bincount(flatseg.flatten())
    top = numpy.argsort(-bc)
    # In a multilabel segmentation, we can't draw everything.
    # Draw the fewest-pixel labels last.  (We could pick the opposite order.)
    for label in top:
        if label == 0:
            continue
        if bc[label] == 0:
            break
        bitmap = ((flatseg == label).sum(axis=0) > 0)
        result[bitmap] = high_contrast_arr[label % len(high_contrast_arr)]
    result = result.reshape((seg.shape[1], seg.shape[2], 3))
    if seg.shape[1:] != size:
        result = scipy.misc.imresize(result, size, interp='nearest')
    return result

# A palette that maximizes perceptual contrast between entries.
# https://stackoverflow.com/questions/33295120
high_contrast = [
    [0, 0, 0], [255, 255, 0], [28, 230, 255], [255, 52, 255],
    [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89],
    [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172],
    [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135],
    [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0],
    [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128],
    [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160],
    [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0],
    [221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153],
    [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111],
    [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191],
    [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9],
    [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255],
    [69, 109, 117], [183, 123, 104], [122, 135, 161], [120, 141, 102],
    [136, 85, 120], [250, 208, 159], [255, 138, 154], [209, 87, 160],
    [190, 196, 89], [69, 102, 72], [0, 134, 237], [136, 111, 76],
    [52, 54, 45], [180, 168, 189], [0, 166, 170], [69, 44, 44],
    [99, 99, 117], [163, 200, 201], [255, 145, 63], [147, 138, 129],
    [87, 83, 41], [0, 254, 207], [176, 91, 111], [140, 208, 255],
    [59, 151, 0], [4, 247, 87], [200, 161, 161], [30, 110, 0],
    [121, 0, 215], [167, 117, 0], [99, 103, 169], [160, 88, 55],
    [107, 0, 44], [119, 38, 0], [215, 144, 255], [155, 151, 0],
    [84, 158, 121], [255, 246, 159], [32, 22, 37], [114, 65, 143],
    [188, 35, 255], [153, 173, 192], [58, 36, 101], [146, 35, 41],
    [91, 69, 52], [253, 232, 220], [64, 78, 85], [0, 137, 163],
    [203, 126, 152], [164, 232, 4], [50, 78, 114], [106, 58, 76],
    [131, 171, 88], [0, 28, 30], [209, 247, 206], [0, 75, 40],
    [200, 208, 246], [163, 164, 137], [128, 108, 102], [34, 40, 0],
    [191, 86, 80], [232, 48, 0], [102, 121, 109], [218, 0, 124],
    [255, 26, 89], [138, 219, 180], [30, 2, 0], [91, 78, 81],
    [200, 149, 197], [50, 0, 51], [255, 104, 50], [102, 225, 211],
    [207, 205, 172], [208, 172, 148], [126, 211, 121], [1, 44, 88],
    [122, 123, 255], [214, 142, 1], [53, 51, 57], [120, 175, 161],
    [254, 178, 198], [117, 121, 124], [131, 115, 147], [148, 58, 77],
    [181, 244, 255], [210, 220, 213], [149, 86, 189], [106, 113, 74],
    [0, 19, 37], [2, 82, 95], [10, 163, 247], [233, 129, 118],
    [219, 213, 221], [94, 188, 209], [61, 79, 68], [126, 100, 5],
    [2, 104, 78], [150, 43, 117], [141, 133, 70], [150, 149, 197],
    [231, 115, 206], [216, 106, 120], [62, 137, 190], [202, 131, 78],
    [81, 138, 135], [91, 17, 60], [85, 129, 59], [231, 4, 196],
    [0, 0, 95], [169, 115, 153], [75, 129, 96], [89, 115, 138],
    [255, 93, 167], [247, 201, 191], [100, 49, 39], [81, 58, 1],
    [107, 148, 170], [81, 160, 88], [164, 91, 2], [29, 23, 2],
    [226, 0, 39], [231, 171, 99], [76, 96, 1], [156, 105, 102],
    [100, 84, 123], [151, 151, 158], [0, 106, 102], [57, 20, 6],
    [244, 215, 73], [0, 69, 210], [0, 108, 49], [221, 182, 208],
    [124, 101, 113], [159, 178, 164], [0, 216, 145], [21, 160, 138],
    [188, 101, 233], [255, 255, 254], [198, 220, 153], [32, 59, 60],
    [103, 17, 144], [107, 58, 100], [245, 225, 255], [255, 160, 242],
    [204, 170, 53], [55, 69, 39], [139, 180, 0], [121, 120, 104],
    [198, 0, 90], [59, 0, 10], [200, 98, 64], [41, 96, 124],
    [64, 35, 52], [125, 90, 68], [204, 184, 124], [184, 129, 131],
    [170, 81, 153], [181, 214, 195], [163, 132, 105], [159, 148, 240],
    [167, 69, 113], [184, 148, 166], [113, 187, 140], [0, 180, 51],
    [120, 158, 201], [109, 128, 186], [149, 63, 0], [94, 255, 3],
    [228, 255, 252], [27, 225, 119], [188, 177, 229], [118, 145, 47],
    [0, 49, 9], [0, 96, 205], [210, 0, 150], [137, 85, 99],
    [41, 32, 29], [91, 50, 19], [167, 111, 66], [137, 65, 46],
    [26, 58, 42], [73, 75, 90], [168, 140, 133], [244, 171, 170],
    [163, 243, 171], [0, 198, 200], [234, 139, 102], [149, 138, 159],
    [189, 201, 210], [159, 160, 100], [190, 71, 0], [101, 129, 136],
    [131, 164, 133], [69, 60, 35], [71, 103, 93], [58, 63, 0],
    [6, 18, 3], [223, 251, 113], [134, 142, 126], [152, 208, 88],
    [108, 143, 125], [215, 191, 194], [60, 62, 110], [216, 61, 102],
    [47, 93, 155], [108, 94, 70], [210, 91, 136], [91, 101, 108],
    [0, 181, 127], [84, 92, 70], [134, 96, 151], [54, 93, 37],
    [37, 47, 153], [0, 204, 255], [103, 78, 96], [252, 0, 156],
    [146, 137, 107], [30, 35, 36], [222, 201, 178], [157, 73, 72],
    [133, 171, 180], [52, 33, 66], [208, 150, 133], [164, 172, 172],
    [0, 255, 255], [174, 156, 134], [116, 42, 51], [14, 114, 197],
    [175, 216, 236], [192, 100, 185], [145, 2, 140], [254, 237, 191],
    [255, 183, 137], [156, 184, 228], [175, 255, 209], [42, 54, 76],
    [79, 74, 67], [100, 112, 149], [52, 187, 255], [128, 119, 129],
    [146, 0, 3], [179, 165, 167], [1, 134, 21], [241, 255, 200],
    [151, 111, 92], [255, 59, 193], [255, 95, 107], [7, 125, 132],
    [245, 109, 147], [87, 113, 218], [78, 30, 42], [131, 0, 85],
    [2, 211, 70], [190, 69, 45], [0, 144, 94], [190, 0, 40],
    [110, 150, 227], [0, 118, 153], [254, 201, 109], [156, 106, 125],
    [63, 161, 184], [137, 61, 227], [121, 180, 214], [127, 212, 217],
    [103, 81, 187], [178, 141, 45], [226, 122, 5], [221, 156, 184],
    [170, 188, 122], [152, 0, 52], [86, 26, 2], [143, 127, 0],
    [99, 80, 0], [205, 125, 174], [138, 94, 45], [255, 179, 225],
    [107, 100, 102], [198, 211, 0], [1, 0, 226], [136, 236, 105],
    [143, 204, 190], [33, 0, 28], [81, 31, 77], [227, 246, 227],
    [255, 142, 177], [107, 79, 41], [163, 127, 70], [106, 89, 80],
    [31, 42, 26], [4, 120, 77], [16, 24, 53], [230, 224, 208],
    [255, 116, 254], [0, 164, 95], [143, 93, 248], [75, 0, 89],
    [65, 47, 35], [216, 147, 158], [219, 157, 114], [96, 65, 67],
    [181, 186, 206], [152, 158, 183], [210, 196, 219], [165, 135, 175],
    [119, 215, 150], [127, 140, 148], [255, 155, 3], [85, 81, 150],
    [49, 221, 174], [116, 182, 113], [128, 38, 71], [42, 55, 63],
    [1, 74, 104], [105, 102, 40], [76, 123, 109], [0, 44, 39],
    [122, 69, 34], [59, 88, 89], [229, 211, 129], [255, 243, 255],
    [103, 159, 160], [38, 19, 0], [44, 87, 66], [145, 49, 175],
    [175, 93, 136], [199, 112, 106], [97, 171, 31], [140, 242, 212],
    [197, 217, 184], [159, 255, 251], [191, 69, 204], [73, 57, 65],
    [134, 59, 96], [185, 0, 118], [0, 49, 119], [197, 130, 210],
    [193, 179, 148], [96, 43, 112], [136, 120, 104], [186, 191, 176],
    [3, 0, 18], [209, 172, 254], [127, 222, 254], [75, 92, 113],
    [163, 160, 151], [230, 109, 83], [99, 123, 93], [146, 190, 165],
    [0, 248, 179], [190, 221, 255], [61, 181, 167], [221, 50, 72],
    [182, 228, 222], [66, 119, 69], [89, 140, 90], [185, 76, 89],
    [129, 129, 213], [148, 136, 139], [254, 214, 189], [83, 109, 49],
    [110, 255, 146], [228, 232, 255], [32, 226, 0], [255, 208, 242],
    [76, 131, 161], [189, 115, 34], [145, 92, 78], [140, 71, 135],
    [2, 81, 23], [162, 170, 69], [45, 27, 33], [169, 221, 176],
    [255, 79, 120], [82, 133, 0], [0, 154, 46], [23, 252, 228],
    [113, 85, 90], [82, 93, 130], [0, 25, 90], [150, 120, 116],
    [85, 85, 88], [11, 33, 44], [30, 32, 43], [239, 191, 196],
    [111, 151, 85], [111, 117, 134], [80, 29, 29], [55, 45, 0],
    [116, 29, 22], [94, 179, 147], [181, 180, 0], [221, 74, 56],
    [54, 61, 255], [173, 101, 82], [102, 53, 175], [131, 107, 186],
    [152, 170, 127], [70, 72, 54], [50, 44, 62], [124, 185, 186],
    [91, 105, 101], [112, 125, 61], [122, 0, 29], [110, 70, 54],
    [68, 58, 56], [174, 129, 255], [72, 144, 121], [137, 115, 52],
    [0, 144, 135], [218, 113, 60], [54, 22, 24], [255, 111, 1],
    [0, 102, 121], [55, 14, 119], [75, 58, 131], [201, 226, 230],
    [196, 65, 112], [255, 69, 38], [115, 190, 84], [196, 223, 114],
    [173, 255, 96], [0, 68, 125], [220, 206, 201], [189, 148, 121],
    [101, 110, 91], [236, 82, 0], [255, 110, 194], [122, 97, 126],
    [221, 174, 162], [119, 131, 127], [165, 51, 39], [96, 142, 255],
    [181, 153, 215], [165, 1, 73], [78, 0, 37], [201, 177, 169],
    [3, 145, 154], [27, 42, 37], [229, 0, 241], [152, 46, 11],
    [182, 113, 128], [224, 88, 89], [0, 96, 57], [87, 143, 155],
    [48, 82, 48], [206, 147, 76], [179, 194, 190], [192, 186, 192],
    [181, 6, 211], [23, 12, 16], [76, 83, 79], [34, 68, 81],
    [62, 65, 65], [120, 114, 109], [182, 96, 43], [32, 4, 65],
    [221, 181, 136], [73, 114, 0], [197, 170, 182], [3, 60, 97],
    [113, 178, 245], [169, 224, 136], [73, 121, 176], [162, 195, 223],
    [120, 65, 73], [45, 43, 23], [62, 14, 47], [87, 52, 76],
    [0, 145, 190], [228, 81, 209], [75, 75, 106], [92, 1, 26],
    [124, 128, 96], [255, 148, 145], [76, 50, 93], [0, 92, 139],
    [229, 253, 164], [104, 209, 182], [3, 38, 65], [20, 0, 35],
    [134, 131, 169], [207, 255, 0], [167, 44, 62], [52, 71, 90],
    [177, 187, 154], [180, 160, 79], [141, 145, 142], [161, 104, 166],
    [129, 61, 58], [66, 82, 24], [218, 131, 134], [119, 97, 51],
    [86, 57, 48], [132, 152, 174], [144, 193, 211], [181, 102, 107],
    [155, 88, 94], [133, 100, 101], [173, 124, 144], [226, 188, 0],
    [227, 170, 224], [178, 194, 254], [253, 0, 57], [0, 155, 117],
    [255, 244, 109], [232, 126, 172], [223, 227, 230], [132, 133, 144],
    [170, 146, 151], [131, 161, 147], [87, 121, 119], [62, 113, 88],
    [198, 66, 137], [234, 0, 114], [196, 168, 203], [85, 200, 153],
    [231, 143, 207], [0, 69, 71], [246, 226, 227], [150, 103, 22],
    [55, 143, 219], [67, 94, 106], [218, 0, 4], [27, 0, 15],
    [91, 156, 143], [110, 43, 82], [1, 17, 21], [227, 232, 196],
    [174, 59, 133], [234, 28, 169], [255, 158, 107], [69, 125, 139],
    [146, 103, 139], [0, 205, 187], [156, 204, 4], [0, 46, 56],
    [150, 197, 127], [207, 246, 180], [73, 40, 24], [118, 110, 82],
    [32, 55, 14], [227, 209, 159], [46, 60, 48], [178, 234, 206],
    [243, 189, 164], [162, 78, 61], [151, 111, 217], [140, 159, 168],
    [124, 43, 115], [78, 95, 55], [93, 84, 98], [144, 149, 111],
    [106, 167, 118], [219, 203, 246], [218, 113, 255], [152, 124, 149],
    [82, 50, 60], [187, 60, 66], [88, 77, 57], [79, 193, 95],
    [162, 185, 193], [121, 219, 33], [29, 89, 88], [189, 116, 78],
    [22, 11, 0], [32, 34, 26], [107, 130, 149], [0, 224, 228],
    [16, 36, 1], [27, 120, 42], [218, 169, 181], [176, 65, 93],
    [133, 146, 83], [151, 160, 148], [6, 227, 196], [71, 104, 140],
    [124, 103, 85], [7, 92, 0], [117, 96, 213], [125, 159, 0],
    [195, 109, 150], [77, 145, 62], [95, 66, 118], [252, 228, 200],
    [48, 48, 82], [79, 56, 27], [229, 165, 50], [112, 102, 144],
    [170, 154, 146], [35, 115, 99], [115, 1, 62], [255, 144, 121],
    [167, 154, 116], [2, 155, 219], [255, 1, 105], [199, 210, 231],
    [202, 136, 105], [128, 255, 205], [187, 31, 105], [144, 176, 171],
    [125, 116, 169], [252, 199, 219], [153, 55, 91], [0, 171, 77],
    [171, 174, 209], [190, 157, 145], [230, 229, 167], [51, 44, 34],
    [221, 88, 123], [245, 255, 247], [93, 48, 51], [109, 56, 0],
    [255, 0, 32], [181, 123, 179], [215, 255, 230], [197, 53, 169],
    [38, 0, 9], [106, 135, 129], [168, 171, 180], [212, 82, 98],
    [121, 75, 97], [70, 33, 178], [141, 164, 219], [199, 200, 144],
    [111, 233, 173], [162, 67, 167], [178, 176, 129], [24, 27, 0],
    [40, 97, 84], [76, 164, 59], [106, 149, 115], [168, 68, 29],
    [92, 114, 123], [115, 134, 113], [208, 207, 203], [137, 123, 119],
    [31, 63, 34], [65, 69, 167], [218, 152, 148], [161, 117, 122],
    [99, 36, 60], [173, 170, 255], [0, 205, 226], [221, 188, 98],
    [105, 142, 177], [32, 132, 98], [0, 183, 224], [97, 74, 68],
    [155, 187, 87], [122, 92, 84], [133, 122, 80], [118, 107, 126],
    [1, 72, 51], [255, 131, 71], [122, 142, 186], [39, 71, 64],
    [148, 100, 68], [235, 216, 230], [100, 98, 65], [55, 57, 23],
    [106, 212, 80], [129, 129, 123], [212, 153, 227], [151, 148, 64],
    [1, 26, 18], [82, 101, 84], [181, 136, 92], [164, 153, 165],
    [3, 173, 137], [179, 0, 139], [227, 196, 181], [150, 83, 31],
    [134, 113, 117], [116, 86, 158], [97, 125, 159], [231, 4, 82],
    [6, 126, 175], [166, 151, 182], [183, 135, 168], [156, 255, 147],
    [49, 29, 25], [58, 148, 89], [110, 116, 110], [176, 197, 174],
    [132, 237, 247], [237, 52, 136], [117, 76, 120], [56, 70, 68],
    [199, 132, 123], [0, 182, 197], [127, 166, 112], [193, 175, 158],
    [42, 127, 255], [114, 165, 140], [255, 192, 127], [157, 235, 221],
    [217, 124, 142], [126, 124, 147], [98, 230, 116], [181, 99, 158],
    [255, 168, 97], [194, 165, 128], [141, 156, 131], [183, 5, 70],
    [55, 43, 46], [0, 152, 255], [152, 89, 117], [32, 32, 76],
    [255, 108, 96], [68, 80, 131], [133, 2, 170], [114, 54, 31],
    [150, 118, 163], [72, 68, 73], [206, 214, 194], [59, 22, 74],
    [204, 167, 99], [44, 127, 119], [2, 34, 123], [163, 126, 111],
    [205, 230, 220], [205, 255, 251], [190, 129, 26], [247, 113, 131],
    [237, 230, 226], [205, 198, 180], [255, 224, 158], [58, 114, 113],
    [255, 123, 89], [78, 78, 1], [74, 198, 132], [139, 200, 145],
    [188, 138, 150], [207, 99, 83], [220, 222, 92], [94, 170, 221],
    [246, 160, 173], [226, 105, 170], [163, 218, 228], [67, 110, 131],
    [0, 46, 23], [236, 251, 255], [161, 194, 182], [80, 0, 63],
    [113, 105, 91], [103, 196, 187], [83, 110, 255], [93, 90, 72],
    [137, 0, 57], [150, 147, 129], [55, 21, 33], [94, 70, 101],
    [170, 98, 195], [141, 111, 129], [44, 97, 53], [65, 6, 1],
    [86, 70, 32], [230, 144, 52], [109, 166, 189], [229, 142, 86],
    [227, 166, 139], [72, 177, 118], [210, 125, 103], [181, 178, 104],
    [127, 132, 39], [255, 132, 230], [67, 87, 64], [234, 228, 8],
    [244, 245, 255], [50, 88, 0], [75, 107, 165], [173, 206, 255],
    [155, 138, 204], [136, 81, 56], [88, 117, 193], [126, 115, 17],
    [254, 165, 202], [159, 139, 91], [165, 91, 84], [137, 0, 106],
    [175, 117, 111], [42, 32, 0], [116, 153, 161], [255, 181, 80],
    [0, 1, 30], [209, 81, 28], [104, 129, 81], [188, 144, 138],
    [120, 200, 235], [133, 2, 255], [72, 61, 48], [196, 34, 33],
    [94, 167, 255], [120, 87, 21], [12, 234, 145], [255, 250, 237],
    [179, 175, 157], [62, 61, 82], [90, 155, 194], [156, 47, 144],
    [141, 87, 0], [173, 215, 156], [0, 118, 139], [51, 125, 0],
    [197, 151, 0], [49, 86, 220], [148, 69, 117], [236, 255, 220],
    [210, 76, 178], [151, 112, 60], [76, 37, 127], [158, 3, 102],
    [136, 255, 236], [181, 100, 129], [57, 109, 43], [86, 115, 95],
    [152, 131, 118], [155, 177, 149], [169, 121, 92], [228, 197, 211],
    [159, 79, 103], [30, 43, 57], [102, 67, 39], [175, 206, 120],
    [50, 46, 223], [134, 180, 135], [194, 48, 0], [171, 232, 107],
    [150, 101, 109], [37, 14, 53], [166, 0, 25], [0, 128, 207],
    [202, 239, 255], [50, 63, 97], [164, 73, 220], [106, 157, 59],
    [255, 90, 228], [99, 106, 1], [209, 108, 218], [115, 96, 96],
    [255, 186, 173], [211, 105, 180], [255, 222, 214], [108, 109, 116],
    [146, 125, 94], [132, 93, 112], [91, 98, 193], [47, 74, 54],
    [228, 95, 53], [255, 59, 83], [172, 132, 221], [118, 41, 136],
    [112, 236, 152], [64, 133, 67], [44, 53, 51], [46, 24, 45],
    [50, 57, 37], [25, 24, 27], [47, 46, 44], [2, 60, 50],
    [155, 158, 226], [88, 175, 173], [92, 66, 77], [122, 197, 166],
    [104, 93, 117], [185, 188, 189], [131, 67, 87], [26, 123, 66],
    [46, 87, 170], [229, 81, 153], [49, 110, 71], [205, 0, 197],
    [106, 0, 77], [127, 187, 236], [243, 86, 145], [215, 197, 74],
    [98, 172, 183], [203, 161, 188], [162, 138, 154], [108, 63, 59],
    [255, 228, 125], [220, 186, 227], [95, 129, 109], [58, 64, 74],
    [125, 191, 50], [230, 236, 220], [133, 44, 25], [40, 83, 102],
    [184, 203, 156], [14, 13, 0], [75, 93, 86], [107, 84, 63],
    [226, 113, 114], [5, 104, 236], [46, 181, 0], [210, 22, 86],
    [239, 175, 255], [104, 32, 33], [45, 32, 17], [218, 76, 255],
    [112, 150, 142], [255, 123, 125], [74, 25, 48], [232, 194, 130],
    [231, 219, 188], [166, 132, 134], [31, 38, 60], [54, 87, 78],
    [82, 206, 121], [173, 170, 169], [138, 159, 69], [101, 66, 210],
    [0, 251, 140], [93, 105, 123], [204, 210, 127], [148, 165, 161],
    [121, 2, 41], [227, 131, 230], [126, 164, 193], [78, 68, 82],
    [75, 44, 0], [98, 11, 112], [49, 76, 30], [135, 74, 166],
    [227, 0, 145], [102, 70, 10], [235, 154, 139], [234, 195, 163],
    [152, 234, 179], [171, 145, 128], [184, 85, 47], [26, 43, 47],
    [148, 221, 197], [157, 140, 118], [156, 131, 51], [148, 169, 201],
    [57, 41, 53], [140, 103, 94], [204, 233, 58], [145, 113, 0],
    [1, 64, 11], [68, 152, 150], [28, 163, 112], [224, 141, 167],
    [139, 74, 78], [102, 119, 118], [70, 146, 173], [103, 189, 168],
    [105, 37, 92], [211, 191, 255], [74, 81, 50], [126, 146, 133],
    [119, 115, 60], [231, 160, 204], [81, 162, 136], [44, 101, 106],
    [77, 92, 94], [201, 64, 58], [221, 215, 243], [0, 88, 68],
    [180, 162, 0], [72, 143, 105], [133, 129, 130], [212, 233, 185],
    [61, 115, 151], [202, 232, 206], [214, 0, 52], [170, 103, 70],
    [158, 85, 133], [186, 98, 0]
]

high_contrast_arr = numpy.array(high_contrast, dtype=numpy.uint8)
