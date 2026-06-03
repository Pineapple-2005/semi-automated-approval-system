import {
  Controller,
  Get,
  Post,
  Param,
  Body,
  UsePipes,
  ValidationPipe,
  ParseUUIDPipe,
  // UseGuards,   // Uncomment when wiring auth
} from '@nestjs/common';
import { VerificationService } from './verification.service';
import { ReviewActionDto } from './dto/review-action.dto';

/**
 * Admin-only controller for the document review queue.
 *
 * SECURITY: All routes here must be protected by your authentication guard.
 * Uncomment @UseGuards(JwtAuthGuard) (or your equivalent) on the class or
 * individual routes, and import the guard from your auth module.
 *
 * Example:
 *   import { JwtAuthGuard } from '../auth/jwt-auth.guard';
 *   @UseGuards(JwtAuthGuard)
 */
@Controller('admin/verification')
// @UseGuards(JwtAuthGuard)  // <-- ADD YOUR AUTH GUARD HERE
@UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
export class VerificationAdminController {
  constructor(private readonly verificationService: VerificationService) {}

  /**
   * GET /admin/verification/queue
   * Returns all documents with status=needs_review for this system.
   * Each item includes extracted OCR data, flags, and a presigned image URL
   * so your admin panel can display the uploaded document alongside the review form.
   */
  @Get('queue')
  async getReviewQueue() {
    return this.verificationService.getReviewQueue();
  }

  /**
   * POST /admin/verification/review/:jobId
   * Approve or reject a flagged document.
   * Body: { action: "approve" | "reject", comment?: string, reviewedBy?: string }
   * Triggers a webhook to the submitting system's callback URL when present.
   */
  @Post('review/:jobId')
  async reviewDocument(
    @Param('jobId', new ParseUUIDPipe()) jobId: string,
    @Body() dto: ReviewActionDto,
  ) {
    return this.verificationService.reviewDocument(jobId, dto);
  }
}
