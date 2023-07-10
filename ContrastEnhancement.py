from numpy import ceil, floor, mean, std, zeros, shape, exp, negative, add, ndarray, max as maxx, min as minn, reshape
from numpy import concatenate, transpose, tile, repeat, divide


def rgb2gray_weights(rgb_image, red_weight, green_weight, blue_weight):
    if red_weight + green_weight + blue_weight == 1:
        new_image = add(add(rgb_image[:, :, 0] * red_weight, rgb_image[:, :, 1] * green_weight),
                        rgb_image[:, :, 2] * blue_weight)
        new_image = new_image / max(new_image)
        return new_image
    else:
        print("the sum of the weights does not equal 1")


def window_maker(row, col, image_shape=(0, 0), window_size=75):
    # window_size must be an odd integer, image_shape should be input as np.shape(array)
    if window_size > ceil(0.4 * image_shape[0]) or window_size > ceil(
                    0.4 * image_shape[1]) or window_size % 2 != 1 \
            or type(window_size) is not type(1):
        # Check to see that the parameters fill the conditions
        print("please input valid parameters")
        if window_size > ceil(0.4 * image_shape[0]):
            print("window size is too big with respect to image_shape(height)")
        if window_size > ceil(0.4 * image_shape[1]):
            print("window size is too big with respect to image_shape(width)")
        if window_size % 2 != 1:
            print("window size must be an odd integer")
        if type(window_size) is not type(1):
            print("window size must be an odd integer")
    else:
        if row - floor(window_size / 2) < 0:
            start_row = 0
            end_row = row + floor(window_size / 2)
            if col - floor(window_size / 2) < 0:
                start_col = 0
                end_col = col + floor(window_size / 2)
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

            elif col + floor(window_size / 2) > image_shape[1]:
                start_col = col - floor(window_size / 2)
                end_col = image_shape[1]
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

            else:
                start_col = col - floor(window_size / 2)
                end_col = col + floor(window_size / 2)
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

        elif row + floor(window_size) > image_shape[0]:
            start_row = row - floor(window_size / 2)
            end_row = image_shape[0]
            if col - floor(window_size / 2) < 0:
                start_col = 0
                end_col = col + floor(window_size / 2)
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

            elif col + floor(window_size / 2) > image_shape[1]:
                start_col = col - floor(window_size / 2)
                end_col = image_shape[1]
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

            else:
                start_col = col - floor(window_size / 2)
                end_col = col + floor(window_size / 2)
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

        else:
            start_row = row - floor(window_size / 2)
            end_row = row + floor(window_size / 2)
            if col - floor(window_size / 2) < 0:
                start_col = 0
                end_col = col + floor(window_size / 2)
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

            elif col + floor(window_size / 2) > image_shape[1]:
                start_col = col - floor(window_size / 2)
                end_col = image_shape[1]
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result

            else:
                start_col = col - floor(window_size / 2)
                end_col = col + floor(window_size / 2)
                result = (int(start_row), int(end_row), int(start_col), int(end_col))
                return result


def find_window_mean(window):
    w_mean = mean(window)
    w_std = std(window)
    return w_mean, w_std


def neighborhood_mean(gray_image, win_size=75):
    gray_image_window_mean = []
    gray_image_window_std = []

    for row in range(0, shape(gray_image)[0]):
        for col in range(0, shape(gray_image)[1]):
            start_row, end_row, start_col, end_col = window_maker(row, col, shape(gray_image), win_size)
            window = gray_image[start_row:end_row, start_col:end_col]
            window_mean, window_std = find_window_mean(window)
            window_mean = [window_mean]
            window_std = [window_std]
            gray_image_window_mean = gray_image_window_mean + window_mean
            gray_image_window_std = gray_image_window_std + window_std

    gray_image_window_mean = ndarray(gray_image_window_mean)
    gray_image_window_std = ndarray(gray_image_window_std)
    gray_image_window_mean = reshape(gray_image_window_mean, shape(gray_image))
    gray_image_window_std = reshape(gray_image_window_std, shape(gray_image))

    return gray_image_window_mean, gray_image_window_std


def __resize_neighborhood_fast(array, target_shape, speed_row=20, speed_col=20):

    remainder_cols = target_shape[1] % speed_col
    array_repeated_cols = repeat(array[:, :-1], speed_col, axis=1)
    array_repeated_cols_last = transpose(tile(array[:, -1], (remainder_cols, 1)))
    array2 = concatenate((array_repeated_cols,array_repeated_cols_last), axis=1)

    remainder_rows = target_shape[0] % speed_row
    array_repeated_rows = repeat(array2[:-1, :], speed_row, axis=0)
    array_repeated_rows_last = tile(array2[-1, :], (remainder_rows,1))
    result = concatenate((array_repeated_rows, array_repeated_rows_last), axis=0)

    return result


def neighborhood_mean_fast(gray_image, speed_row=20, speed_col=20, win_size=75):
    gray_image_window_mean = []
    gray_image_window_std = []
    row_temp = 0
    col_temp = 0

    for row in range(0, shape(gray_image)[0], speed_row):
        for col in range(0, shape(gray_image)[1], speed_col):
            start_row, end_row, start_col, end_col = window_maker(row, col, shape(gray_image), win_size)
            window = gray_image[start_row:end_row, start_col:end_col]
            window_mean, window_std = find_window_mean(window)
            window_mean = [window_mean]
            window_std = [window_std]
            gray_image_window_mean = gray_image_window_mean + window_mean
            gray_image_window_std = gray_image_window_std + window_std
            col_temp = col
            row_temp = row

    shape_row_temp = int(row_temp / speed_row) + 1
    shape_col_temp = int(col_temp / speed_col) + 1

    gray_image_window_mean = ndarray(gray_image_window_mean)
    gray_image_window_std = ndarray(gray_image_window_std)
    gray_image_window_mean = reshape(gray_image_window_mean, (shape_row_temp, shape_col_temp))
    gray_image_window_std = reshape(gray_image_window_std, (shape_row_temp, shape_col_temp))

    gray_image_window_mean_resize = __resize_neighborhood_fast(gray_image_window_mean, shape(gray_image),
                                                               speed_row, speed_col)
    gray_image_window_std_resize = __resize_neighborhood_fast(gray_image_window_std, shape(gray_image),
                                                              speed_row, speed_col)
    result = (gray_image_window_mean, gray_image_window_std, gray_image_window_mean_resize,
              gray_image_window_std_resize)

    return result


def __contrast_enhancement_nint(gray_image, t, factor):
    result = zeros(shape=shape(gray_image))
    for row in range(shape(gray_image)[0]):
        for col in range(shape(gray_image)[1]):
            result[row, col] = 1 / (1 + exp(-t * (gray_image[row, col] - factor)))
    return __linearly_normalize_image(result)


def contrast_enhancement_nint_multi(gray_image, t=[], factor=[]):
    result = []
    if len(t) > 0 and len(factor) > 0:
        for count1 in range(len(t)):
            for count2 in range(len(factor)):
                curr_t = t[count1]
                curr_factor = factor[count2]
                result.append(__contrast_enhancement_nint(gray_image, curr_t, curr_factor))
    else:
        pass
    return result


def __gaussian(value, height, img_mean, img_std):
    result = height * exp(negative((value - img_mean) ** 2 / (2 * (img_std ** 2))))
    return result


def __linearly_normalize_image(gray_image):
    max_val = maxx(gray_image)
    min_val = minn(gray_image)
    m = 1 / (max_val - min_val)
    b = 1 - m
    new_image = zeros(shape=shape(gray_image))
    for row in range(shape(gray_image)[0]):
        for col in range(shape(gray_image)[1]):
            pixel = gray_image[row, col]
            pixel_new = (m * pixel) + b
            new_image[row, col] = pixel_new
    return new_image


def gaussian_ce(gray_image, height, img_mean, img_std):
    new_image = zeros(shape=shape(gray_image))
    for row in range(shape(gray_image)[0]):
        for col in range(shape(gray_image)[1]):
            if gray_image[row, col] <= img_mean:
                pixel = gray_image[row, col]
                pixel_new = __gaussian(pixel, height, img_mean, img_std)
                new_image[row, col] = pixel_new
            else:
                pixel_new = height
                new_image[row, col] = pixel_new
    new_image = __linearly_normalize_image(new_image)
    return new_image


def root_ce(gray_image, root):
    new_image = zeros(shape=shape(gray_image))
    for row in range(shape(gray_image)[0]):
        for col in range(shape(gray_image)[1]):
            pixel = gray_image[row, col]
            pixel_new = pixel ** (1 / root)
            new_image[row, col] = pixel_new
    return new_image
