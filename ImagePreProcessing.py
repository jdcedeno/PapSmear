from skimage.color import rgb2gray
import numpy as np


class ImagePreProcessing:
    def __init__(self, image):
        self.image = image
        self.original_image = image / 255
        # -----------------------------------           DEBUGGING           ----------------------------------- #
        # Verify that the input image is a numpy ndarray with the assert 'simple statement':
        assert (type(self.original_image) is type(np.zeros(shape=(1, 1, 3))))
        # ----------------------------------------------------------------------------------------------------- #
        # -----------------------------------        Some values       ---------------------------------------- #
        self.shape = self.original_image.shape
        self.size = self.original_image.size
        # ----------------------------------------------------------------------------------------------------- #
        # -----------------------------------        Gray Image       ----------------------------------------- #
        self.gray_image = rgb2gray(image)
        self.gray_image_shape = self.gray_image.shape
        self.gray_image_size = self.gray_image.size
        # ----------------------------------------------------------------------------------------------------- #
        # ---------------------------        Red, Green, and Blue Images       -------------------------------- #
        self.red_image = image[:, :, 0] / 255
        self.green_image = image[:, :, 1] / 255
        self.blue_image = image[:, :, 2] / 255
        # ----------------------------------------------------------------------------------------------------- #
        # --------------------       Contrast Enhanced Image using the NINT operator        ------------------- #
        self.contrast_enhanced_image_nint = []
        self.contrast_enhanced_image_nint_multi = []
        # ----------------------------------------------------------------------------------------------------- #
        # --------------------       Local Contrast Enhancement on very dark areas        --------------------- #
        self.contrast_enhanced_image_dark_places = []
        self.gray_image_window_mean = []
        self.gray_image_window_std = []
        self.speed_row = 20
        self.speed_col = 20

        # ----------------------------------------------------------------------------------------------------- #
    # -------------------------------------        Class Methods       ---------------------------------------- #
    @staticmethod
    def __window_maker(row, col, image_shape=(0, 0), window_size=75):
        # window_size must be an odd integer, image_shape should be input as np.shape(array)
        if window_size > np.ceil(0.4 * image_shape[0]) or window_size > np.ceil(
                        0.4 * image_shape[1]) or window_size % 2 != 1 \
                or type(window_size) is not type(1):
            # Check to see that the parameters fill the conditions
            print("please input valid parameters")
            if window_size > np.ceil(0.4 * image_shape[0]):
                print("window size is too big with respect to image_shape(height)")
            if window_size > np.ceil(0.4 * image_shape[1]):
                print("window size is too big with respect to image_shape(width)")
            if window_size % 2 != 1:
                print("window size must be an odd integer")
            if type(window_size) is not type(1):
                print("window size must be an odd integer")
        else:
            if row - np.floor(window_size / 2) < 0:
                start_row = 0
                end_row = row + np.floor(window_size / 2)
                if col - np.floor(window_size / 2) < 0:
                    start_col = 0
                    end_col = col + np.floor(window_size / 2)
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

                elif col + np.floor(window_size / 2) > image_shape[1]:
                    start_col = col - np.floor(window_size / 2)
                    end_col = image_shape[1]
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

                else:
                    start_col = col - np.floor(window_size / 2)
                    end_col = col + np.floor(window_size / 2)
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

            elif row + np.floor(window_size) > image_shape[0]:
                start_row = row - np.floor(window_size / 2)
                end_row = image_shape[0]
                if col - np.floor(window_size / 2) < 0:
                    start_col = 0
                    end_col = col + np.floor(window_size / 2)
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

                elif col + np.floor(window_size / 2) > image_shape[1]:
                    start_col = col - np.floor(window_size / 2)
                    end_col = image_shape[1]
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

                else:
                    start_col = col - np.floor(window_size / 2)
                    end_col = col + np.floor(window_size / 2)
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

            else:
                start_row = row - np.floor(window_size / 2)
                end_row = row + np.floor(window_size / 2)
                if col - np.floor(window_size / 2) < 0:
                    start_col = 0
                    end_col = col + np.floor(window_size / 2)
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

                elif col + np.floor(window_size / 2) > image_shape[1]:
                    start_col = col - np.floor(window_size / 2)
                    end_col = image_shape[1]
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

                else:
                    start_col = col - np.floor(window_size / 2)
                    end_col = col + np.floor(window_size / 2)
                    result = (int(start_row), int(end_row), int(start_col), int(end_col))
                    return result

    @staticmethod
    def __find_window_mean(window):
        mean = np.mean(window)
        std = np.std(window)
        return mean, std

    @staticmethod
    def __resize_neighborhood_fast(array, target_shape, speed_row=20, speed_col=20):

        remainder_cols = target_shape[1] % speed_col
        array_repeated_cols = np.repeat(array[:,:-1], speed_col, axis=1)
        array_repeated_cols_last = np.transpose(np.tile(array[:,-1],(remainder_cols,1)))
        array2 = np.concatenate((array_repeated_cols,array_repeated_cols_last),axis=1)

        remainder_rows = target_shape[0] % speed_row
        array_repeated_rows = np.repeat(array2[:-1,:],speed_row,axis=0)
        array_repeated_rows_last = np.tile(array2[-1,:],(remainder_rows,1))
        result = np.concatenate((array_repeated_rows,array_repeated_rows_last),axis=0)

        return result

    def red_channel(self):
        red_image = np.zeros(shape=self.shape)
        red_image[0:, 0:, 0] = self.image[0:, 0:, 0] / 255
        return red_image

    def green_channel(self):
        green_image = np.zeros(shape=self.shape)
        green_image[0:, 0:, 1] = self.image[0:, 0:, 1] / 255
        return green_image

    def blue_channel(self):
        blue_image = np.zeros(shape=self.shape)
        blue_image[0:, 0:, 2] = self.image[0:, 0:, 2] / 255
        return blue_image

    def __contrast_enhancement_nint(self, t, factor):
        result = np.zeros(shape=self.gray_image_shape)
        for row in range(self.gray_image_shape[0]):
            for col in range(self.gray_image_shape[1]):
                result[row, col] = 1 / (1 + np.exp(-t * (self.gray_image[row, col] - factor)))
        self.contrast_enhanced_image_nint = result
        return result

    def contrast_enhancement_nint_multi(self, t=[], factor=[]):
        self.contrast_enhanced_image_nint_multi = []
        if len(t) > 0 and len(factor) > 0:
            for count1 in range(len(t)):
                for count2 in range(len(factor)):
                    curr_t = t[count1]
                    curr_factor = factor[count2]
                    self.contrast_enhanced_image_nint_multi.append(self.contrast_enhancement_nint(curr_t, curr_factor))
        else:
            pass
        result = self.contrast_enhanced_image_nint_multi
        return result

    def neighborhood_mean(self, win_size=75):
        gray_image_window_mean = []
        gray_image_window_std = []

        for row in range(0, self.gray_image_shape[0]):
            for col in range(0, self.gray_image_shape[1]):
                start_row, end_row, start_col, end_col = self.__window_maker(row, col, self.gray_image_shape, win_size)
                window = self.gray_image[start_row:end_row, start_col:end_col]
                window_mean, window_std = self.__find_window_mean(window)
                window_mean = [window_mean]
                window_std = [window_std]
                gray_image_window_mean = gray_image_window_mean + window_mean
                gray_image_window_std = gray_image_window_std + window_std

        gray_image_window_mean = np.array(gray_image_window_mean)
        gray_image_window_std = np.array(gray_image_window_std)
        gray_image_window_mean = np.reshape(gray_image_window_mean, self.gray_image_shape)
        gray_image_window_std = np.reshape(gray_image_window_std, self.gray_image_shape)

        return gray_image_window_mean, gray_image_window_std

    def neighborhood_mean_fast(self, speed_row=20, speed_col=20, win_size=75):
        gray_image_window_mean = []
        gray_image_window_std = []
        row_temp = 0
        col_temp = 0
        self.speed_row = speed_row
        self.speed_col = speed_col

        for row in range(0, self.gray_image_shape[0], speed_row):
            for col in range(0, self.gray_image_shape[1], speed_col):
                start_row, end_row, start_col, end_col = self.__window_maker(row, col, self.gray_image_shape, win_size)
                window = self.gray_image[start_row:end_row, start_col:end_col]
                window_mean, window_std = self.__find_window_mean(window)
                window_mean = [window_mean]
                window_std = [window_std]
                gray_image_window_mean = gray_image_window_mean + window_mean
                gray_image_window_std = gray_image_window_std + window_std
                col_temp = col
                row_temp = row

        shape_row_temp = int(row_temp / speed_row) + 1
        shape_col_temp = int(col_temp / speed_col) + 1

        gray_image_window_mean = np.array(gray_image_window_mean)
        gray_image_window_std = np.array(gray_image_window_std)
        gray_image_window_mean = np.reshape(gray_image_window_mean, (shape_row_temp, shape_col_temp))
        gray_image_window_std = np.reshape(gray_image_window_std, (shape_row_temp, shape_col_temp))

        gray_image_window_mean_resize = self.__resize_neighborhood_fast(gray_image_window_mean, self.gray_image_shape,
                                                                         speed_row, speed_col)
        gray_image_window_std_resize = self.__resize_neighborhood_fast(gray_image_window_std, self.gray_image_shape,
                                                                         speed_row, speed_col)
        result = (gray_image_window_mean, gray_image_window_std, gray_image_window_mean_resize,
                  gray_image_window_std_resize)

        return result

    @staticmethod
    def __gaussian(value, height, mean, std):
        result = height * np.exp(np.negative((value - mean) ** 2 / (2 * (std ** 2))))
        return result

    @staticmethod
    def __linearly_normalize_image(gray_image):
        max_val = np.max(gray_image)
        min_val = np.min(gray_image)
        m = 1 / (max_val - min_val)
        b = 1 - m
        new_image = np.zeros(shape=np.shape(gray_image))
        for row in range(np.shape(gray_image)[0]):
            for col in range(np.shape(gray_image)[1]):
                pixel = gray_image[row, col]
                pixel_new = (m * pixel) + b
                new_image[row, col] = pixel_new
        return new_image

    def gaussian_ce(self, image, height, mean, std):
        new_image = np.zeros(shape=np.shape(image))
        for row in range(np.shape(image)[0]):
            for col in range(np.shape(image)[1]):
                if image[row, col] <= mean:
                    pixel = image[row, col]
                    pixel_new = self.__gaussian(pixel, height, mean, std)
                    new_image[row, col] = pixel_new
                else:
                    pixel_new = height
                    new_image[row, col] = pixel_new
        new_image = self.__linearly_normalize_image(new_image)
        return new_image

    @staticmethod
    def euler_ce(image):
        new_image = np.zeros(shape=np.shape(image))
        for row in range(np.shape(image)[0]):
            for col in range(np.shape(image)[1]):
                pixel = image[row, col]
                pixel_new = np.exp(pixel)
                new_image[row, col] = pixel_new
        return new_image

    @staticmethod
    def root_ce(image, root):
        new_image = np.zeros(shape=np.shape(image))
        for row in range(np.shape(image)[0]):
            for col in range(np.shape(image)[1]):
                pixel = image[row, col]
                pixel_new = pixel ** (1 / root)
                new_image[row, col] = pixel_new
        return new_image

    @staticmethod
    def rgb2gray_weights(image, red_weight, green_weight, blue_weight):
        if red_weight + green_weight + blue_weight == 1:
            new_image = np.add(np.add(image[:, :, 0] * red_weight, image[:, :, 1] * green_weight),
                               image[:, :, 2] * blue_weight)
            new_image = new_image / np.max(new_image)
            return new_image
        else:
            print("the sum of the weights does not equal 1")