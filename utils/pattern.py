import pickle
import cv2
import imutils


def find_valley_peak_pattern(peaks, valleys, peak_tolerance=10):
    data = list(
        sorted(
            list(zip(peaks, ["peak"] * len(peaks)))
            + list(zip(valleys, ["valley"] * len(valleys))),
            key=lambda x: x[0][0],
        )
    )
    
    patterns = []
    n = len(data)

    i = 0
    while i < n - 4:
        # Extract consecutive groups of 5 dots
        dot1, dot2, dot3, dot4, dot5 = data[i : i + 5]

        # Check for valley-peak-valley-peak-valley pattern
        if (
            # dot1[1] == "valley" and
            dot2[1] == "peak"
            and dot3[1] == "valley"
            and dot4[1] == "peak"
            # and dot5[1] == "valley"
        ):
            y1, y2, y3, y4, y5 = (
                dot1[0][1],
                dot2[0][1],
                dot3[0][1],
                dot4[0][1],
                dot5[0][1],
            )

            # Check if the two peaks have approximately the same y-coordinate
            if abs(y2 - y4) <= peak_tolerance:
                # Check if the first and last valleys are lower than the middle valley
                if y1 > y3 - peak_tolerance and y5 > y3 - peak_tolerance:
                    patterns.append((dot1, dot2, dot3, dot4, dot5))
                    # Skip to the next group after the current pattern
                    i += 5
                    continue

        # Move to the next dot
        i += 1

    return patterns


def plot_pattern(
    image,
    pattern,
    peak_line_color=(0, 255, 0),
    horizontal_line_color=(0, 0, 255),
    line_thickness=2,
):
    valley1, peak1, valley2, peak2, valley3 = pattern

    # Extract coordinates
    valley1_coords = valley1[0]
    peak1_coords = peak1[0]
    valley2_coords = valley2[0]
    peak2_coords = peak2[0]
    valley3_coords = valley3[0]

    # Horizontal line connecting the peaks
    peak_y = min(
        peak1_coords[1], peak2_coords[1]
    )  # Use the smaller y-coordinate of the two peaks
    peak_center_x = (peak1_coords[0] + peak2_coords[0]) // 2
    peak_line_half_width = abs(
        peak1_coords[0] - peak2_coords[0]
    )  # Half width = distance between peaks
    cv2.line(
        image,
        (peak_center_x - peak_line_half_width, peak_y),
        (peak_center_x + peak_line_half_width, peak_y),
        peak_line_color,
        thickness=line_thickness,
    )

    # Horizontal line between valley1 and valley3 at valley2's y-coordinate
    valley_y = valley2_coords[1]
    cv2.line(
        image,
        (valley1_coords[0], valley_y),
        (valley3_coords[0], valley_y),
        horizontal_line_color,
        thickness=line_thickness,
    )

    return image


def main():
    with open("coord.pickle", "rb") as file:
        peaks, valleys = pickle.load(file)

    image = cv2.imread(r".\testing\Screenshot 2024-12-01 232716.png")
    # image = cv2.imread(args["image"])
    image = imutils.resize(image, width=min(640, image.shape[1]))

    for x, y in peaks:
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in valleys:
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Image with Dots", image)

    patterns = find_valley_peak_pattern(peaks, valleys, peak_tolerance=50)
    print(patterns)
    for pattern in patterns:
        image = plot_pattern(image, pattern)
    cv2.imshow("Image with Patterns", image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
