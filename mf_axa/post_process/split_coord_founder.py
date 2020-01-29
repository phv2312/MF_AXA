import time
import cv2
import numpy as np
import skimage.measure as measure

class SplitCoordFounder:
    def __init__(self):
        self.PAD = 50
        self.THRESHOLD_AREA_COMPONENT = 50
        self.LOWER_RATIO_LINE_POS = 0.2
        self.UPPER_RATIO_LINE_POS = 0.6
        self.LINE_POS_STEP = 3
        self.MAX_NOISE_PER_LINE = 20

    def process(self, multi_form_image, direction = None):
        assert direction in ['h', 'w', None], 'unknown direction !!!'

        if direction is not None:
            try:
                row_or_col_id, area =  self._process(multi_form_image, direction)
                return row_or_col_id, area, direction
            except Exception as e:
                print ('Ex:', e)

                return -1, -1, direction
        else:
            col, col_area, _ = self.process(multi_form_image, direction='w')
            row, row_area, _ = self.process(multi_form_image, direction='h')

            if col_area > row_area:  # w
                return col, col_area, 'w'
            else:  # h
                return row, row_area, 'h'


    def _process(self, multi_form_image, direction):
        """
        :param multi_form_image: input must be cv2 np.ndarray, multiple form image.
        :param direction: 'h' or 'w', if 'h', multiple-forms in horizontal direction, otherwise vertical direction.
        :return:
        """
        # background black, white foreground
        cv2_im = multi_form_image.copy()
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)
        cv2_im = cv2.bitwise_not(cv2_im)
        cv2_im = cv2_im[self.PAD:-self.PAD, self.PAD:-self.PAD]

        # threshold & pre-processing
        cv2_im = cv2.threshold(cv2_im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2_im = cv2.morphologyEx(cv2_im, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

        # remove small object
        new_im = cv2.erode(cv2_im, np.ones((3,3), dtype=np.uint8))

        # finding candidate splitting lines by counting the number of white pixel each hoz/ver line.
        # some codes may be duplicated because of the laziness ...
        h, w = new_im.shape[:2]
        copy_im = np.zeros_like(new_im)

        selected_ids = []
        if direction == 'h':
            row_sum = np.sum(new_im, axis=1)

            _start, _end = int(self.LOWER_RATIO_LINE_POS * h), int(self.UPPER_RATIO_LINE_POS * h)

            for row_id in range(_start, _end, self.LINE_POS_STEP):
                _sum = row_sum[row_id]
                if _sum > self.MAX_NOISE_PER_LINE: continue

                _area1 = sum(row_sum[:row_id])
                _area2 = sum(row_sum[row_id:])

                max_v = max(_area1, _area2)
                min_v = min(_area1, _area2)

                ratio = max_v / (min_v + 1e-6)
                if 1. <= ratio <= 6.:
                    selected_ids += [row_id]
                    cv2.line(copy_im, (0, row_id), (w, row_id), 127, thickness=1)

            copy_im = cv2.dilate(copy_im, kernel=np.ones(shape=(self.LINE_POS_STEP, 1), dtype=np.uint8))

            #
            label = measure.label(copy_im, neighbors=8, background=0)
            selected_region = None
            for region in measure.regionprops(label):
                if selected_region is None: selected_region = region
                else:
                    if selected_region.area < region.area:
                        selected_region = region

            row, col = selected_region.centroid
            return int(row) + self.PAD, selected_region.area

        elif direction == 'w':
            col_sum = np.sum(new_im, axis=0) # (w)

            _start, _end = int(self.LOWER_RATIO_LINE_POS * w), int(self.UPPER_RATIO_LINE_POS * w)

            for col_id in range(_start, _end,self.LINE_POS_STEP):
                _sum = col_sum[col_id]
                if _sum > self.MAX_NOISE_PER_LINE: continue

                _area1 = sum(col_sum[:col_id])
                _area2 = sum(col_sum[col_id:])

                max_v = max(_area1, _area2)
                min_v = min(_area1, _area2)

                ratio = max_v / (min_v + 1e-6)
                if 1. <= ratio <= 5.:
                    selected_ids += [col_id]
                    cv2.line(copy_im, (col_id, 0), (col_id, h), 127, thickness=1)

            copy_im = cv2.dilate(copy_im, kernel=np.ones(shape=(1, self.LINE_POS_STEP * 2), dtype=np.uint8))

            #
            label = measure.label(copy_im, neighbors=8, background=0)
            selected_region = None
            for region in measure.regionprops(label):
                if selected_region is None:
                    selected_region = region
                else:
                    if selected_region.area < region.area:
                        selected_region = region

            row, col = selected_region.centroid
            return int(col) + self.PAD, selected_region.area

        else:
            raise Exception('unknown direction')

def test3_1():
    import glob, os
    model = SplitCoordFounder()

    sample_im_dir = "/home/kan/Desktop/split_data (1)/multiple_form/train/special"
    output_dir = os.path.join(sample_im_dir, 'result')
    os.makedirs(output_dir, exist_ok=True)

    for sample_im_fn in glob.glob(os.path.join(sample_im_dir, '*.tif')):
        sample_im = cv2.imread(sample_im_fn)

        s_time = time.time()
        col_or_row_id, area, direction = model.process(sample_im, direction=None)
        if direction == 'w':
            col = col_or_row_id
            if col > -1:
                cv2.line(sample_im, (col, 0), (col, sample_im.shape[0]), (255, 0, 0), thickness=10)
            else:
                print("Cannot split")
        elif direction == 'h':
            row = col_or_row_id
            if row > -1:
                cv2.line(sample_im, (0, row), (sample_im.shape[1], row), (255, 0, 0), thickness=10)
            else:
                print("Cannot split")

        print('take time:', time.time() - s_time)
        out_fn = os.path.basename(sample_im_fn)
        cv2.imwrite(os.path.join(output_dir, out_fn), sample_im)

if __name__ == '__main__':
    test3_1()

