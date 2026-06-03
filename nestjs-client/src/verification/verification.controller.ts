import {
  Controller,
  Post,
  Get,
  Param,
  Body,
  UploadedFile,
  UseInterceptors,
  UsePipes,
  ValidationPipe,
  ParseUUIDPipe,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { VerificationService } from './verification.service';
import { SubmitDocumentDto } from './dto/submit-document.dto';

/**
 * Handles user-facing document submission and status polling.
 * Mount this controller at /verification in your app's module.
 */
@Controller('verification')
@UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
export class VerificationController {
  constructor(private readonly verificationService: VerificationService) {}

  /**
   * POST /verification
   * Upload an ID document for verification.
   * Expects multipart/form-data with fields from SubmitDocumentDto + file field "file".
   * Returns {jobId, status: "processing", message} immediately.
   */
  @Post()
  @UseInterceptors(FileInterceptor('file'))
  async submitDocument(
    @UploadedFile() file: Express.Multer.File,
    @Body() dto: SubmitDocumentDto,
  ) {
    return this.verificationService.submitDocument(file, dto);
  }

  /**
   * GET /verification/status/:jobId
   * Poll processing status. Returns full job data including extracted fields,
   * flags, and a presigned image URL when complete.
   */
  @Get('status/:jobId')
  async getStatus(@Param('jobId', new ParseUUIDPipe()) jobId: string) {
    return this.verificationService.getJobStatus(jobId);
  }
}
