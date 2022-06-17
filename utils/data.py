import torch

# def collate_fn(batch):
#     return tuple(zip(*batch))


# custom collate function, just to show the idea
def collate_fn(batch):
    data = [item[0] for item in batch]
    target = []
    for item in batch:
        labels = item[1]
        target.append(
            {
                "boxes": torch.FloatTensor(labels["boxes"]),
                "labels": torch.LongTensor(labels["labels"]),
                "image_id": torch.LongTensor(labels["image_id"]),
                "area": torch.FloatTensor(labels["area"]),
                "iscrowd": torch.IntTensor(labels["iscrowd"]),
            }
        )
    return [data, target]


# custom collate function, just to show the idea
# def collate_data(batch, device):
#     data = [item[0].to(device) for item in batch]
#     target = []
#     for item in batch:
#         labels = item[1]
#         target.append(
#             {
#                 "boxes": torch.FloatTensor(labels["boxes"]).to(device),
#                 "labels": torch.LongTensor(labels["labels"]).to(device),
#                 "image_id": torch.LongTensor(labels["image_id"]).to(device),
#                 "area": torch.FloatTensor(labels["area"]).to(device),
#                 "iscrowd": torch.IntTensor(labels["iscrowd"]).to(device),
#             }
#         )
#     return [data, target]


# def collate_data(batch):
#     return tuple(zip(*batch))
