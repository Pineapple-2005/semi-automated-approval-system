import { IsString, IsIn, IsOptional, IsUrl, IsNotEmpty } from 'class-validator';

export class SubmitDocumentDto {
  @IsString()
  @IsNotEmpty()
  userId: string;

  @IsIn(['umid', 'passport', 'drivers_license', 'philsys', 'prc'], {
    message: 'documentType must be one of: umid, passport, drivers_license, philsys, prc',
  })
  documentType: string;

  @IsString()
  @IsNotEmpty()
  expectedName: string;

  @IsOptional()
  @IsString()
  expectedDob?: string;

  @IsOptional()
  @IsUrl({}, { message: 'callbackUrl must be a valid URL' })
  callbackUrl?: string;
}
