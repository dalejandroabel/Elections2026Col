import cv2
import numpy as np
import pypdfium2 as pdfium
import base64
import json
from dotenv import load_dotenv
import os
import threading


load_dotenv()

FEATURES_DIR = os.getenv("FEATURES_DIR")
cv2.setUseOptimized(True)
import threading

_FEATURE_CACHE: dict | None = None
_FEATURE_LOCK = threading.Lock()

def _load_features(features_dir: str) -> dict:
    global _FEATURE_CACHE
    if _FEATURE_CACHE is not None:
        return _FEATURE_CACHE
    with _FEATURE_LOCK:
        if _FEATURE_CACHE is not None: 
            return _FEATURE_CACHE
        sift = cv2.SIFT_create()
        cache = {
            "votes":      cv2.imread(os.path.join(features_dir, "vote_feature.png"),      cv2.IMREAD_GRAYSCALE),
            "c1":         cv2.imread(os.path.join(features_dir, "c1_features.png"),        cv2.IMREAD_GRAYSCALE),
            "c2":         cv2.imread(os.path.join(features_dir, "c2_features.png"),        cv2.IMREAD_GRAYSCALE),
            "nivelation": cv2.imread(os.path.join(features_dir, "nivelation_feature.png"), cv2.IMREAD_GRAYSCALE),
            "total":      cv2.imread(os.path.join(features_dir, "total_feature.png"),      cv2.IMREAD_GRAYSCALE),
        }
        missing = [k for k, v in cache.items() if v is None]
        if missing:
            raise FileNotFoundError(f"Feature images not found: {missing}")
        for key, img in cache.items():
            kp, des = sift.detectAndCompute(img, None)
            cache[key] = {"img": img, "kp": kp, "des": des}
        _FEATURE_CACHE = cache   # single atomic assignment — only happens once
    return _FEATURE_CACHE


class E14Extractor():


    def __init__(self, src, canvass_type="V", render_scale=3, verbose=False, custom_limits = {}):
        feats = _load_features(FEATURES_DIR)
        self.votes_feature = feats["votes"]["img"]
        self.c1_feature = feats["c1"]["img"]
        self.c2_feature = feats["c2"]["img"]
        self.nivelation_feature = feats["nivelation"]["img"]
        self.total_feature = feats["total"]["img"]

        self.canvass_type = canvass_type
        self.set_sections_limits(custom_limits)
        self.set_cell_size()
        self.src = src
        self.render_scale = render_scale
        self.images, self.gray_images = self.get_images_from_src()
        self.verbose = verbose
        self.logs = []
        self.treshold_value = 127

    def set_sections_limits(self, custom_limits=None):
        if self.canvass_type == "V":
            squares_x = [0.73, 0.985]
            self.page_lims = custom_limits.get("page", {"x": [0, 1], "y": [0.225, 0.96]})
            self.nivelation_lims = custom_limits.get("nivelation", {"x": squares_x, "y": [0.2, 0.95]})
            self.candidates_lims = custom_limits.get("candidates", {"x": squares_x, "y": [1/3, 2/3]})
            self.total_count_lims = custom_limits.get("total_count", {"x": squares_x, "y": [0.05, 0.95]})
        else:
            pass

    def change_limits(self, limit_name, new_lims):
        if hasattr(self, f"{limit_name}_lims"):
            setattr(self, f"{limit_name}_lims", new_lims)
        else:
            raise ValueError(f"Invalid limit name: {limit_name}")

    def set_cell_size(self,):
        if self.canvass_type == "V":
            self.cell_size = [75, 75]
        else:
            pass

    @property
    def canvass_type(self):
        return self._canvass_type

    @canvass_type.setter
    def canvass_type(self, canvass_type):
        if canvass_type not in ["V", "J", "T"]:
            raise ValueError("canvass_type must be 'V', 'J', or 'T'")
        self._canvass_type = canvass_type

    def crop_image(self, image, lims={"x": [0, 1], "y": [0, 1]}):
        """ Crop an image according to the given limits

        Args:
            image (np.ndarray): The input image to crop
            lims (dict, optional): The limits for cropping, this are fractions of the image dimensions. Defaults to {"x": [0, 1], "y": [0, 1]}.

        Returns:
            np.ndarray: The region of the image defined by the limits
        """
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        h, w = image.shape[:2]
        x1, x2 = int(w*lims["x"][0]), int(w*lims["x"][1])
        y1, y2 = int(h*lims["y"][0]), int(h*lims["y"][1])
        return image[y1:y2, x1:x2]

    def display(self, image, scale=1):
        """ Display the image in a new window

        Args:
            image (np.ndarray): The image to display
            scale (int, optional): The scale factor for resizing the image. Defaults to 1.
        """
        if image.size > 0:
            cv2.imshow('Image', cv2.resize(
                image, (int(image.shape[1]*scale), int(image.shape[0]*scale))))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def display_contours(self, image, contours, prop=1, index=-1):
        """ Display the image with the contours drawn on it

        Args:
            image (np.ndarray): The image to display
            contours (list): The list of contours to draw
            prop (int, optional): The scale factor for resizing the image. Defaults to 1.
            index (int, optional): The index of the contour to draw. Defaults to -1.
        """
        c_image = image.copy()
        cv2.drawContours(c_image, contours, index, (0, 255, 0), 2)
        self.display(c_image, scale=prop)

    def get_images_from_src(self):
        """ Get the images from the source PDF as bytes and convert them to numpy arrays

        Returns:
            tuple: A tuple containing the cropped color images and the cropped grayscale images
        """
        pdf = pdfium.PdfDocument(self.src)
        try:
            images = [
            pdf[i].render(scale=self.render_scale).to_numpy()
            for i in range(min(2, len(pdf)))
        ]
        finally:
            pdf.close()  # always release the native handle
        img_c, img_g = [], []
        for img in images:
            c = self.crop_image(img, self.page_lims)
            img_c.append(c)
            img_g.append(cv2.cvtColor(
                np.ascontiguousarray(c), cv2.COLOR_RGB2GRAY))
        return img_c, img_g

    def treshold_image(self, image, treshold=None, bimodal=True):
        """ Treshold the image using Otsu's method or a fixed treshold

        Args:
            image (np.ndarray): The image to threshold
            treshold (int, optional): The threshold value. Defaults to None.
            bimodal (bool, optional): Whether to use a bimodal histogram. Defaults to True.

        Returns:
            np.ndarray: The thresholded image
        """
        if treshold is None:
            treshold = self.treshold_value
        if bimodal:
            return cv2.threshold(image, treshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return cv2.threshold(image, treshold, 255, cv2.THRESH_BINARY_INV)[1]

    def sort_contours(self, contours):
        """ Sort contours by their vertical position (top to bottom)

        Args:
            contours (list): The list of contours to sort

        Returns:
            list: The sorted list of contours
        """
        moments = [cv2.moments(c) for c in contours]
        cy = [m['m01'] / m['m00'] if m['m00'] != 0 else 0 for m in moments]
        return [c for _, c in sorted(zip(cy, contours))]
    
    def get_valid_contours(self, image, contours, min_ratio = 5, max_ratio = 20):
        """  Filter contours by their area ratio to the image area

        Args:
            image (np.ndarray): The image containing the contours
            contours (list[np.ndarray]): The list of contours to filter
            min_ratio (int, optional): The minimum area ratio. Defaults to 5.
            max_ratio (int, optional): The maximum area ratio. Defaults to 20.

        Returns:
            list[np.ndarray]: The list of contours that have an area ratio between the minimum and maximum values
        """
        image_area = image.shape[0] * image.shape[1]
        return [c for c in contours if min_ratio < (100*cv2.contourArea(c)/image_area) < max_ratio]

    def get_all_contours(self, image, expected_n=8):
        """ Get all contours from the image, filter them by area ratio, sort them by vertical position, and optionally pop a contour by index

        Args:
            image (np.ndarray): The image containing the contours
            pop_contour (bool, optional): Whether to pop a contour. Defaults to True.
            pop_index (int, optional): The index of the contour to pop. Defaults to 1.
            expected_n (int, optional): The expected number of contours. Defaults to 8.

        Returns:
            list[np.ndarray]: The list of filtered and sorted contours
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = self.get_valid_contours(image, contours)
        if len(contours) != expected_n:
            raise ValueError(
                f"Expected {expected_n} contours, but found {len(contours)}")
        contours = self.sort_contours(contours)

        return contours

    def crop_contour(self, image, contour, lims):
        """ Crop the region of the image defined by the contour and the limits

        Args:
            image (np.ndarray): The image containing the region to crop
            contour (np.ndarray): The contour defining the region to crop
            lims (tuple): The limits for cropping, this are fractions of the contour bounding box dimensions.

        Returns:
            np.ndarray: The cropped image region
        """
        x, y, w, h = cv2.boundingRect(contour)
        return self.crop_image(image[y:y+h, x:x+w], lims)

    def get_feature_image(self, src):
        return cv2.imread(src, cv2.IMREAD_GRAYSCALE)

    def match_feature(self, image, feature_name, threshold_match=0.8, min_match_count=10, draw = False):
        
        feat = _FEATURE_CACHE[feature_name]
        kpf, desf = feat["kp"], feat["des"]
        sift = cv2.SIFT_create()
        kpim, desim = sift.detectAndCompute(image,None)
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desf,desim,k=2)
        valid_matches = [m for m,n in matches if m.distance < threshold_match*n.distance]

        if len(valid_matches)>min_match_count:
            src_pts = np.float32([ kpf[m.queryIdx].pt for m in valid_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kpim[m.trainIdx].pt for m in valid_matches ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = feat["img"].shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
        else:
            raise ValueError(f"Not enough matches are found - {len(valid_matches)}/{min_match_count}")
        
        if draw:
            draw_image = cv2.polylines(image.copy(), [np.int32(dst)], True, 50, 3, cv2.LINE_AA)
            draw_params = dict(matchColor = (0,255,0),singlePointColor = None,
                        matchesMask = matchesMask, flags = 2)
            match_images = cv2.drawMatches(feat["img"],kpf,draw_image,kpim,valid_matches,None,**draw_params)  
            return dst, match_images
        return dst   

    def get_feature_center(self, dst):
        return np.mean(np.array(dst).reshape(-1, 2), axis=0)  
    
    def get_candidates_distance(self, dst_c1, dst_c2):
        center_c1 = self.get_feature_center(dst_c1)
        center_c2 = self.get_feature_center(dst_c2)
        return abs(center_c1[1] - center_c2[1])
    
    def get_vote_to_candidate_distance(self, dst_vote, first_candidate_dst):
        center_vote = self.get_feature_center(dst_vote)
        center_candidate = self.get_feature_center(first_candidate_dst)
        return abs(center_vote[1] - center_candidate[1])

    def get_features_relations(self, image):
        votes_dst = self.match_feature(image, "votes")
        c1_dst = self.match_feature(image, "c1")
        c2_dst = self.match_feature(image, "c2")

        dvc = self.get_vote_to_candidate_distance(votes_dst, c1_dst)
        dy_candidates = self.get_candidates_distance(c1_dst, c2_dst)

        return dvc, dy_candidates

    def get_candidates_centers(self, center_vote, dvc, dc, n_candidates=7):
        dy = dc
        centers = []
        for i in range(n_candidates):
            center_candidate = [center_vote[0], center_vote[1] + dvc + i*dy]
            centers.append(center_candidate)
        return centers
    
    def get_candidate_squares(self, image, candidate_centers, dx_square, dy_square):
        candidate_squares = []
        for center in candidate_centers:
            square = image[int(center[1]-dy_square/2):int(center[1]+dy_square/2),
                        int(center[0]-dx_square/2):int(center[0]+dx_square/2)]
            candidate_squares.append(square)
        return candidate_squares

    def resolve_first_page_by_features(self, image, dvc, dy_candidates):
        center_vote = self.get_feature_center(self.match_feature(image, "votes"))
        dx_squares = self.votes_feature.shape[1]
        dy_square = int(dx_squares/3)
        candidates_centers = self.get_candidates_centers(center_vote, dvc, dy_candidates, n_candidates=7)
        candidates_squares = self.get_candidate_squares(image, candidates_centers, dx_squares, dy_square)
        dst_n = self.match_feature(image, "nivelation")
        center_n = self.get_feature_center(dst_n)
        nivelation_square = image[int(center_n[1]-3*dy_square/2):int(center_n[1]+3*dy_square/2),
                                int(center_vote[0]-dx_squares/2):int(center_vote[0]+dx_squares/2)]
        return nivelation_square, candidates_squares
    
    def resolve_second_page_by_features(self, image, dvc, dy_candidates):
        center_vote = self.get_feature_center(self.match_feature(image, "votes"))
        dx_squares = self.votes_feature.shape[1]
        dy_square = int(dx_squares/3)
        candidates_centers = self.get_candidates_centers(center_vote, dvc, dy_candidates, n_candidates=6)
        candidates_squares = self.get_candidate_squares(image, candidates_centers, dx_squares, dy_square)
        dst_t = self.match_feature(image, "total")
        center_t = self.get_feature_center(dst_t)
        total_square = image[int(center_t[1]-2*dy_square):int(center_t[1]+2*dy_square),
                            int(center_vote[0]-dx_squares/2):int(center_vote[0]+dx_squares/2)]
        return total_square, candidates_squares

    def resolve_pages_by_features(self):
        dvc, dy_candidates = self.get_features_relations(self.gray_images[0])
        nivelation, candidates1= self.resolve_first_page_by_features(self.gray_images[0], dvc, dy_candidates)
        total, candidates2= self.resolve_second_page_by_features(self.gray_images[1], dvc, dy_candidates)

        return nivelation, total, candidates1 +  candidates2

    def resolve_first_page_by_contours(self, image):
        """ Resolve the first page of the canvass, extracting the nivelation and candidates sections

        Args:
            image (np.ndarray): The image containing the contours to extract

        Returns:
            tuple: A tuple containing the nivelation and candidates sections as numpy arrays
        """
        contours = self.get_all_contours(self.treshold_image(
            image), expected_n=8)
        nivelation = self.crop_contour(
            image, contours.pop(0), self.nivelation_lims)
        candidates = [self.crop_contour(
            image, contour, self.candidates_lims) for contour in contours]
        candidates = [c if not self.is_empty(
            c) else np.array([]) for c in candidates]
        return nivelation, candidates

    def resolve_second_page_by_contours(self, image):
        """ Resolve the second page of the canvass, extracting the total count and candidates sections

        Args:
            image (np.ndarray): The image containing the contours to extract

        Returns:
            tuple: A tuple containing the total count and candidates sections as numpy arrays
        """
        contours = self.get_all_contours(self.treshold_image(
            image), expected_n=7)
        total_count = self.crop_contour(
            image, contours.pop(-1), self.total_count_lims)
        candidates = [self.crop_contour(
            image, contour, self.candidates_lims) for contour in contours]
        candidates = [c if not self.is_empty(
            c) else np.array([]) for c in candidates]
        return total_count, candidates

    def is_empty(self, image):
        """ Check if the image is empty by verifying if all pixel values are above the threshold value

        Args:
            image (np.ndarray): The image to check for emptiness

        Returns:
            bool: True if the image is empty, False otherwise
        """
        return not np.any(image <= self.treshold_value)

    def resolve_pages_by_contours(self):
        """ Resolve both pages of the canvass, extracting the nivelation, total count, and candidates sections using contours
        Returns:
            tuple: A tuple containing the nivelation, total count, and candidates sections as numpy arrays
        """
        nivelation, candidates1 = self.resolve_first_page_by_contours(self.gray_images[0])
        total_count, candidates2 = self.resolve_second_page_by_contours(self.gray_images[1])
        return nivelation, total_count, candidates1 + candidates2

    def resolve_pages(self):
        """ Resolve both pages of the canvass, extracting the nivelation, total count, and candidates sections

        Returns:
            tuple: A tuple containing the nivelation, total count, and candidates sections as numpy arrays
        """
        try: 
            if self.verbose:
                print("Resolving pages by contours...")
            return self.resolve_pages_by_contours()
        except ValueError as e:
            try:
                if self.verbose:
                    print("Failed to resolve pages by contours, trying features...")
                return self.resolve_pages_by_features()
            except ValueError as e:
                raise ValueError("Could not resolve pages by contours or features") from e

    def split_cells(self, image, splits=(1, 1)):
        """ Split the image into cells according to the given splits, the image must be in grayscale

        Args:
            image (np.ndarray): The image to split into cells
            splits (tuple, optional): The number of rows and columns to split the image into. Defaults to (1, 1) (No split).

        Returns:
            np.ndarray: The split cells as a 4D array with shape (rows, cols, cell_height, cell_width)
        """
        h, w = image.shape
        rows, cols = splits

        dy = h // rows
        dx = w // cols

        cells = (
            image[:rows * dy, :cols * dx]
            .reshape(rows, dy, cols, dx)
            .transpose(0, 2, 1, 3)
        )

        cells_resized = np.empty((rows, cols, *self.cell_size), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                cells_resized[r, c] = cv2.resize(
                    cells[r, c],
                    self.cell_size,
                    interpolation=cv2.INTER_AREA
                )
        return cells_resized

    def remove_empty_cells(self, cells):
        """ Remove empty cells from the split cells, returning an empty array for cells that are empty

        Args:
            cells (np.ndarray): The split cells as a 4D array with shape (rows, cols, cell_height, cell_width)

        Returns:
            list: A list of lists containing the non-empty cells, with empty arrays for cells that are empty
        """
        valid_mask = np.any(cells <= self.treshold_value, axis=(2, 3))
        return [
            [
                cells[r, c] if valid_mask[r, c] else []
                for c in range(valid_mask.shape[1])
            ]
            for r in range(valid_mask.shape[0])
        ]
    
    def resolve_as_cells(self):
        nivelation, total_count, candidates = self.resolve_pages()
        n_splits = self.remove_empty_cells(self.split_cells(nivelation, splits=(3, 3)))
        t_splits = self.remove_empty_cells(self.split_cells(total_count, splits=(4, 3)))
        candidates_splits = [self.remove_empty_cells(self.split_cells(candidate, splits=(1, 3))) if candidate.size > 0 else [] for candidate in candidates]
        return n_splits, t_splits, candidates_splits

    def encode_image(self, image):
        if len(image) == 0:
            return ""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode("utf-8")
    
    def resolve_as_json(self):
        nivelation_cells, total_count_cells, candidates_cells = self.resolve_as_cells()
        n_squares = 3
        total_E11 = [self.encode_image(nivelation_cells[0][i]) for i in range(n_squares)]
        total_ballot_box = [self.encode_image(nivelation_cells[1][i]) for i in range(n_squares)]
        total_burned = [self.encode_image(nivelation_cells[2][i]) for i in range(n_squares)]

        blank_votes = [self.encode_image(total_count_cells[0][i]) for i in range(n_squares)]
        null_votes = [self.encode_image(total_count_cells[1][i]) for i in range(n_squares)]
        not_marked_votes = [self.encode_image(total_count_cells[2][i]) for i in range(n_squares)]
        total_sum_votes = [self.encode_image(total_count_cells[3][i]) for i in range(n_squares)]

        candidates_votes = {}
        for i, candidate in enumerate(candidates_cells):
            candidate_key = f"id_{i+1}"
            if len(candidate) == 0:
                candidates_votes[candidate_key] = []
            else:
                candidates_votes[candidate_key] = [self.encode_image(candidate[0][j]) for j in range(n_squares)]

        data = {
            "nivelation": {
                "total_E11": total_E11,
                "total_ballot_box": total_ballot_box,
                "total_burned": total_burned
            },
            "total_votes": {
                "blank": blank_votes,
                "null": null_votes,
                "not_marked": not_marked_votes,
                "total_sum": total_sum_votes
            },
            "candidates_votes": candidates_votes
        }
        return data
    
    def save_as_json(self, filename):
        data = self.resolve_as_json()
        with open(filename, 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    test_url = "https://escrutiniospresidente2026.registraduria.gov.co/docs/E14/01/001/01/09/E14_PRE_01_001_001_01_09_005_5007.pdf"
    import requests
    response = requests.get(test_url)
    content = response.content
    
    EX = E14Extractor(content, canvass_type="V", verbose=True, render_scale = 3)
    # print(EX.images[0].shape)
    # nivelation, total_count, candidates = EX.resolve_pages()

    # EX.display(nivelation, scale=0.5)
