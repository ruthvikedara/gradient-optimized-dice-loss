def dice_coefficient(pred, target, epsilon= 1e-6):
  """
  Calculates the Dice coefficient between predicted and target tensors.
  
  Args:
      pred (torch.Tensor): Predicted tensor of shape (B x C x H x W)
      target (torch.Tensor): Target tensor of shape (B x C x H x W)
      epsilon (float): Small value to avoid division by zero (default: 1e-6)
  
  Returns:
      torch.Tensor: Dice coefficient for each sample in the batch
  """
  intersection = (pred * target).sum(dim=(1,2,3))
  union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
  dice = (2 * intersection) / (intersection + union + epsilon)
  return dice
  
def gradient_optimized_dice_loss(pred, target):
  """
  Calculates the Gradient-Optimized Dice Loss.
  
  Args:
      pred (torch.Tensor): Predicted tensor of shape (B x C x H x W)
      target (torch.Tensor): Target tensor of shape (B x C x H x W)
  
  Returns:
      torch.Tensor: Gradient-Optimized Dice Loss
  """
  batch_size = pred.size(0)
  dice_scores = dice_coefficient(pred, target)

  intersection = (pred * target).sum(dim=(1,2,3))
  union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

  # Calculate the norm of the union vector
  union_norms = torch.linalg.vector_norm(union, dim=0)

  # Dice loss
  dice_loss = 1 - dice_scores
  
  # Mean of union norms
  mean_union = union_norms.mean()
  
  # Normalized union norms
  normalized_union = union_norms / mean_union
  
  # Gradient-Optimized Dice Loss
  godc_loss = dice_loss * normalized_union

  return godc_loss.mean()


# Example usage
if __name__ == "__main__":
    # Example tensors
    pred = torch.randn((1, 1, 256, 256))
    target = torch.randint(0, 2, (1, 1, 256, 256))

    # Calculate loss
    loss = gradient_optimized_dice_loss(pred, target)
    print(f'Gradient-Optimized Dice Loss: {loss.item()}')
  
