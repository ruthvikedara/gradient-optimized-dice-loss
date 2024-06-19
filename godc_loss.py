def dice_coeffecient(pred, target, epsilon= 1e-6):
  """
  Calculates Dice coefficient between predicted and target tensors
  Input: Two tensors of shape (C x H x W)
  Output: 
  """
  intersection = (pred * target).sum(dim=(1,2,3))
  union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
  dice = (2 * intersection) / (intersection + union + epsilon)
  return dice
  
def gradient_optimized_dice_loss(pred, target):
  """
  """
  batch_size = pred.size(0)
  dice_scores = dice_coefficient(pred, target)

  intersection = (pred * target).sum(dim=(1,2,3))
  union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

  # Normalized union vectors
  union_norms = torch.linalg.vector_norm(union, dim=0)

  dice_loss = 1 - dice_scores
  mean_union = union_norms.mean()
  normalized_union = union_norms / mean_union
  godc_loss = dice_loss * normalized_union

  return godc_loss.mean()
  
