import { IsIn, IsOptional, IsString, IsNotEmpty } from 'class-validator';

export class ReviewActionDto {
  @IsIn(['approve', 'reject'], {
    message: 'action must be either "approve" or "reject"',
  })
  action: 'approve' | 'reject';

  @IsOptional()
  @IsString()
  comment?: string;

  @IsOptional()
  @IsString()
  @IsNotEmpty()
  reviewedBy?: string;
}
