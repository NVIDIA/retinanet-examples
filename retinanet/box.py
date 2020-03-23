import torch
from ._C import decode as decode_cuda
from ._C import decode_rotate as decode_rotate_cuda
from ._C import iou as iou_cuda
from ._C import nms as nms_cuda
from ._C import nms_rotate as nms_rotate_cuda
import numpy as np

def order_points(pts):
    pts_reorder = []

    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, 0])
        xSorted = pt[idx, :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[torch.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        (br, tr) = rightMost[torch.argsort(D, descending=True), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))

    return torch.stack([p for p in pts_reorder])


def rotate_boxes(boxes, points=False):
    '''
    Rotate target bounding boxes

    Input:
        Target boxes (xmin_ymin, width_height, theta)
    Output:
        boxes_axis (xmin_ymin, xmax_ymax, theta)
        boxes_rotated (xy0, xy1, xy2, xy3)
    '''

    u = torch.stack([torch.cos(boxes[:, 4]), torch.sin(boxes[:, 4])], dim=1)
    l = torch.stack([-torch.sin(boxes[:, 4]), torch.cos(boxes[:, 4])], dim=1)
    R = torch.stack([u, l], dim=1)

    if points:
        cents = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1).transpose(1, 0)
        boxes_rotated = torch.stack([boxes[:, 0], boxes[:, 1],
                                     boxes[:, 2], boxes[:, 1],
                                     boxes[:, 2], boxes[:, 3],
                                     boxes[:, 0], boxes[:, 3],
                                     boxes[:, -2],
                                     boxes[:, -1]], 1)

    else:
        cents = torch.stack([boxes[:, 0] + (boxes[:, 2] - 1) / 2, boxes[:, 1] + (boxes[:, 3] - 1) / 2], 1).transpose(1,
                                                                                                                     0)
        boxes_rotated = torch.stack([boxes[:, 0], boxes[:, 1],
                                     (boxes[:, 0] + boxes[:, 2] - 1), boxes[:, 1],
                                     (boxes[:, 0] + boxes[:, 2] - 1), (boxes[:, 1] + boxes[:, 3] - 1),
                                     boxes[:, 0], (boxes[:, 1] + boxes[:, 3] - 1),
                                     boxes[:, -2],
                                     boxes[:, -1]], 1)

    xy0R = torch.matmul(R, boxes_rotated[:, :2].transpose(1, 0) - cents) + cents
    xy1R = torch.matmul(R, boxes_rotated[:, 2:4].transpose(1, 0) - cents) + cents
    xy2R = torch.matmul(R, boxes_rotated[:, 4:6].transpose(1, 0) - cents) + cents
    xy3R = torch.matmul(R, boxes_rotated[:, 6:8].transpose(1, 0) - cents) + cents

    xy0R = torch.stack([xy0R[i, :, i] for i in range(xy0R.size(0))])
    xy1R = torch.stack([xy1R[i, :, i] for i in range(xy1R.size(0))])
    xy2R = torch.stack([xy2R[i, :, i] for i in range(xy2R.size(0))])
    xy3R = torch.stack([xy3R[i, :, i] for i in range(xy3R.size(0))])

    boxes_axis = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4] - 1,
                            torch.sin(boxes[:, -1, None]), torch.cos(boxes[:, -1, None])], 1)
    boxes_rotated = order_points(torch.stack([xy0R, xy1R, xy2R, xy3R], dim=1)).view(-1, 8)

    return boxes_axis, boxes_rotated


def generate_anchors(stride, ratio_vals, scales_vals):
    'Generate anchors coordinates from scales/ratios'

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def generate_anchors_rotated(stride, ratio_vals, scales_vals, angles_vals, device):
    'Generate anchors coordinates from scales/ratios/angles'

    scales = torch.FloatTensor(scales_vals).to(device).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals)).to(device)

    wh = torch.FloatTensor([stride]).to(device).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)

    xy1 = 0.5 * (wh - dwh * scales)
    xy3 = 0.5 * (wh + dwh * scales)
    xy2 = xy1 + (xy3 - xy1) * torch.FloatTensor([0, 1]).to(device)
    xy4 = xy1 + (xy3 - xy1) * torch.FloatTensor([1, 0]).to(device)

    anchors = torch.stack([xy1, xy2, xy3, xy4], 1).view(-1, 4, 2)
    angles = torch.FloatTensor(angles_vals).to(device)
    anchors = anchors.repeat(angles.size(0), 1, 1)
    theta = angles.repeat(xy1.size(0), 1).transpose(0, 1).contiguous().view(-1)

    xmin_ymin = xy1.repeat(int(theta.size(0) / xy1.size(0)), 1)
    xmax_ymax = xy3.repeat(int(theta.size(0) / xy3.size(0)), 1)

    u = torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1)
    l = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
    R = torch.stack([u, l], dim=1).to(device)

    cents = torch.FloatTensor([stride / 2]).to(device)
    anchors_axis = torch.cat([xmin_ymin, xmax_ymax], dim=1)
    anchors_rotated = order_points(torch.matmul(anchors - cents, R) + cents).view(-1, 8)

    return anchors_axis, anchors_rotated


def box2delta(boxes, anchors):
    'Convert boxes to deltas from anchors'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat([
        (boxes_ctr - anchors_ctr) / anchors_wh,
        torch.log(boxes_wh / anchors_wh)
    ], 1)


def box2delta_rotated(boxes, anchors):
    'Convert boxes to deltas from anchors'

    anchors_wh = anchors[:, 2:4] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:4] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
    boxes_sin = boxes[:, 4]
    boxes_cos = boxes[:, 5]

    return torch.cat([
        (boxes_ctr - anchors_ctr) / anchors_wh,
        torch.log(boxes_wh / anchors_wh), boxes_sin[:, None], boxes_cos[:, None]
    ], 1)


def delta2box(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1)
    ], 1)


def delta2box_rotated(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:4] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:4]) * anchors_wh
    pred_sin = deltas[:, 4]
    pred_cos = deltas[:, 5]

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1),
        torch.atan2(pred_sin, pred_cos)[:, None]
    ], 1)


def snap_to_anchors(boxes, size, stride, anchors, num_classes, device):
    'Snap target boxes (x, y, w, h) to anchors'

    num_anchors = anchors.size()[0] if anchors is not None else 1
    width, height = (int(size[0] / stride), int(size[1] / stride))

    if boxes.nelement() == 0:
        return (torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device))

    boxes, classes = boxes.split(4, dim=1)

    # Generate anchors
    x, y = torch.meshgrid([torch.arange(0, size[i], stride, device=device, dtype=classes.dtype) for i in range(2)])
    xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4).to(dtype=classes.dtype)
    anchors = (xyxy + anchors).contiguous().view(-1, 4)

    # Compute overlap between boxes and anchors
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], 1)
    xy1 = torch.max(anchors[:, None, :2], boxes[:, :2])
    xy2 = torch.min(anchors[:, None, 2:], boxes[:, 2:])
    inter = torch.prod((xy2 - xy1 + 1).clamp(0), 2)
    boxes_area = torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1)
    anchors_area = torch.prod(anchors[:, 2:] - anchors[:, :2] + 1, 1)
    overlap = inter / (anchors_area[:, None] + boxes_area - inter)

    # Keep best box per anchor
    overlap, indices = overlap.max(1)
    box_target = box2delta(boxes[indices], anchors)
    box_target = box_target.view(num_anchors, 1, width, height, 4)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()

    depth = torch.ones_like(overlap) * -1
    depth[overlap < 0.4] = 0  # background
    depth[overlap >= 0.5] = classes[indices][overlap >= 0.5].squeeze() + 1  # objects
    depth = depth.view(num_anchors, width, height).transpose(1, 2).contiguous()

    # Generate target classes
    cls_target = torch.zeros((anchors.size()[0], num_classes + 1), device=device, dtype=boxes.dtype)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[overlap < 0.4] = num_classes  # background has no class
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()

    return (cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            depth.view(num_anchors, 1, height, width))


def snap_to_anchors_rotated(boxes, size, stride, anchors, num_classes, device):
    'Snap target boxes (x, y, w, h, a) to anchors'

    anchors_axis, anchors_rotated = anchors

    num_anchors = anchors_rotated.size()[0] if anchors_rotated is not None else 1
    width, height = (int(size[0] / stride), int(size[1] / stride))

    if boxes.nelement() == 0:
        return (torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 6, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device))

    boxes, classes = boxes.split(5, dim=1)
    boxes_axis, boxes_rotated = rotate_boxes(boxes)

    # Generate anchors
    x, y = torch.meshgrid([torch.arange(0, size[i], stride, device=device, dtype=classes.dtype) for i in range(2)])
    xy_2corners = torch.stack((x, y, x, y), 2).unsqueeze(0)
    xy_4corners = torch.stack((x, y, x, y, x, y, x, y), 2).unsqueeze(0)
    anchors_axis = (xy_2corners.to(torch.float) + anchors_axis.view(-1, 1, 1, 4)).contiguous().view(-1, 4)
    anchors_rotated = (xy_4corners.to(torch.float) + anchors_rotated.view(-1, 1, 1, 8)).contiguous().view(-1, 8)

    if torch.cuda.is_available():
        iou = iou_cuda

    overlap = iou(boxes_rotated.contiguous().view(-1), anchors_rotated.contiguous().view(-1))[0]

    # Keep best box per anchor
    overlap, indices = overlap.max(1)
    box_target = box2delta_rotated(boxes_axis[indices], anchors_axis)
    box_target = box_target.view(num_anchors, 1, width, height, 6)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()

    depth = torch.ones_like(overlap, device=device) * -1
    depth[overlap < 0.4] = 0  # background
    depth[overlap >= 0.5] = classes[indices][overlap >= 0.5].squeeze() + 1  # objects
    depth = depth.view(num_anchors, width, height).transpose(1, 2).contiguous()

    # Generate target classes
    cls_target = torch.zeros((anchors_axis.size()[0], num_classes + 1), device=device, dtype=boxes_axis.dtype)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[overlap < 0.4] = num_classes  # background has no class
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()

    return (cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 6, height, width),
            depth.view(num_anchors, 1, height, width))


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None):
    'Box Decoding and Filtering'

    if torch.cuda.is_available():
        return decode_cuda(all_cls_head.float(), all_box_head.float(),
                           anchors.view(-1).tolist(), stride, threshold, top_n)

    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, 4), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, 4)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices / width) % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, 4, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride + anchors[a, :]
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.size()[0]] = scores
        out_boxes[batch, :boxes.size()[0], :] = boxes
        out_classes[batch, :classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def decode_rotated(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None):
    'Box Decoding and Filtering'

    if torch.cuda.is_available():
        return decode_rotate_cuda(all_cls_head.float(), all_box_head.float(),
                                  anchors[0].view(-1).tolist(), stride, threshold, top_n)

    device = all_cls_head.device
    anchors = anchors[0].to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, 6), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, 6)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices / width) % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, 6, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride + anchors[a, :]
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.size()[0]] = scores
        out_boxes[batch, :boxes.size()[0], :] = boxes
        out_classes[batch, :classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    'Non Maximum Suppression'

    if torch.cuda.is_available():
        return nms_cuda(
            all_scores.float(), all_boxes.float(), all_classes.float(), nms, ndetections)

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes


def nms_rotated(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    'Non Maximum Suppression'

    if torch.cuda.is_available():
        return nms_rotate_cuda(
            all_scores.float(), all_boxes.float(), all_classes.float(), nms, ndetections)

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 6), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 6)
        classes = all_classes[batch, keep].view(-1)
        theta = torch.atan2(boxes[:, -2], boxes[:, -1])
        boxes_theta = torch.cat([boxes[:, :-2], theta[:, None]], dim=1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, boxes_theta, classes = boxes[indices], boxes_theta[indices], classes[indices]
        areas = (boxes_theta[:, 2] - boxes_theta[:, 0] + 1) * (boxes_theta[:, 3] - boxes_theta[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            boxes_axis, boxes_rotated = rotate_boxes(boxes_theta, points=True)
            overlap, inter = iou(boxes_rotated.contiguous().view(-1), boxes_rotated[i, :].contiguous().view(-1))
            inter = inter.squeeze()
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 6)
            boxes_theta = boxes_theta[criterion.nonzero(), :].view(-1, 5)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes
