import cv2
import numpy as np
from scipy import spatial

from .cv2api import cv2api_delegate

class kdtreeOpticalFlow:
    def __init__(self, prvs_frame, next_frame, flow, bkflow):
        self.prvs_frame = prvs_frame
        self.next_frame = next_frame
        self.flow = flow
        self.bkflow = bkflow
        self.hight = flow.shape[0]
        self.width = flow.shape[1]
        self.count = 0
        h, w = flow.shape[:2]
        self.coords = (np.swapaxes(np.indices((w, h), np.float32), 0, 2))

    def setFrames(self, prvs_frame, next_frame, flow, bkflow):
        self.prvs_frame = prvs_frame
        self.next_frame = next_frame
        self.flow = flow
        self.bkflow = bkflow

    def warpFlow(self, img, flow):
        res = self.adjustFlow_G(flow)
        adj = res[0] + self.coords
        mp = res[1]
        return cv2.remap(img, adj, None, cv2.INTER_LINEAR), mp

    def adjustFlow_G(self, flow, p=3.0, k=5):
        p = p
        k = k
        h, w = flow.shape[:2]
        coord = (np.swapaxes(np.indices((w, h), np.float32), 0, 2))
        cpy = np.copy(flow)
        cpy += coord
        ktree = spatial.cKDTree(np.reshape(cpy, (w * h, 2)))
        reverse = np.swapaxes(np.indices((w, h), np.float32), 0, 2)
        reverse[:][:] = -1000.0
        nearest = ktree.query(coord, k=k)  # ,distance_upper_bound=2.0**0.5)
        mp = np.any((nearest[0] < 1.0), axis=2)
        # identify points that have source points close enough to use
        close_enough = np.any((nearest[0] < 1.0), axis=2)
        # id points that have at least one match 0 away
        exact = np.any((nearest[0] == 0.0), axis=2)
        values = np.asarray([nearest[1] % w, nearest[1] / w])

        for x in range(h):
            for y in range(w):
                found = False
                dist = nearest[0][x, y]
                # skip points too far away
                if not close_enough[x, y]:
                    continue

                # process exact matches
                if exact[x, y]:
                    # find max distance of the k closest source points
                    md_k = np.argmax(((values[1, x, y] - x) ** 2 + (values[0, x, y] - y) ** 2) ** .5)

                    # if mapped distance ==0 and source distance is the greatest, use it
                    if dist[md_k] == 0:
                        reverse[x, y, :] = values[:, x, y, md_k]
                        #                       reverse[x][y][0] = values[0,x,y,md_k]
                        found = True

                # process interpolation points
                if not found:
                    weight_accum = np.sum(1 / dist[np.where(dist[:] > 0)] ** p)
                    w_xyk = np.sum((values[:, x, y, np.where(dist[:] > 0)]) /
                                   (dist[np.where(dist[:] > 0)] ** p), axis=2)
                    reverse[x][y] = (w_xyk / weight_accum)[:, 0]

        return reverse - coord, mp

    def setTime(self, frame_time, truth=None):
        forward_flow = np.multiply(self.flow, 1 - frame_time)
        # cv2.imwrite('fwd_frame.png',imageafy(forward_flow,self.next_frame))
        backward_flow = np.multiply(self.bkflow, frame_time)
        # cv2.imwrite('bwd_frame.png', imageafy(backward_flow, self.next_frame))
        from_prev, mpp = self.warpFlow(self.prvs_frame, backward_flow)
        from_next, mpn = self.warpFlow(self.next_frame, forward_flow)
        # cv2.imwrite('bwd_frame_real' + 'new' + str(self.count) + '.png', from_prev)
        # cv2.imwrite('fwd_frame_real' + 'new' + str(self.count) + '.png', from_next)
        truth = self.next_frame if truth is None else truth
        f = self.frameadjust(from_next, self.prvs_frame, mpp)
        n = self.frameadjust(from_prev, truth, mpn)
        from_next = f
        from_prev = n
        from_prev = np.multiply(from_prev, (1 - frame_time))
        from_next = np.multiply(from_next, frame_time)

        frame = (np.add(from_prev, from_next)).astype(np.uint8)
        self.count += 1
        return frame

    def frameadjust(self, frame, alterframe, mp):
        cpy = np.copy(frame)
        for x in range(len(frame)):
            for y in range(len(frame[0])):
                if np.array_equal(frame[x][y], np.zeros(len(frame[x][y]))):
                    cpy[x][y] = alterframe[x][y]
        return cpy


def interpolate_frames(before_frame, after_frame, num_new_frames):
    fwd_flow, back_flow = get_flow(before_frame, after_frame, 'forward')
    opticalFlow = kdtreeOpticalFlow(before_frame, after_frame, fwd_flow, back_flow)
    interpolated_frames = []
    for i in range(1, int(num_new_frames + 1)):
        frame_scale = i / (1.0 + num_new_frames)
        frame = opticalFlow.setTime(frame_scale)
        interpolated_frames.append(frame)

    return interpolated_frames


def get_flow(before_frame, after_frame, direction):
    if direction != 'forward':
        before_frame_gray = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)
        after_frame_gray = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
    else:
        before_frame_gray = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
        after_frame_gray = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)

    back_flow = cv2api_delegate.calcOpticalFlowFarneback(before_frame_gray, after_frame_gray, 0.8, 7, 15, 3, 7, 1.5, 2)
    fwd_flow = cv2api_delegate.calcOpticalFlowFarneback(after_frame_gray, before_frame_gray, 0.8, 7, 15, 3, 7, 1.5, 2)

    return fwd_flow, back_flow