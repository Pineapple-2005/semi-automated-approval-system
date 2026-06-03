import { Injectable, Logger, HttpException, HttpStatus } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import * as FormData from 'form-data';
import { SubmitDocumentDto } from './dto/submit-document.dto';
import { ReviewActionDto } from './dto/review-action.dto';

@Injectable()
export class VerificationService {
  private readonly logger = new Logger(VerificationService.name);

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {}

  private get baseUrl(): string {
    return this.configService.getOrThrow<string>('VERIFY_API_URL').replace(/\/$/, '');
  }

  private get authHeaders(): Record<string, string> {
    return {
      'X-API-Key': this.configService.getOrThrow<string>('VERIFY_API_KEY'),
    };
  }

  private get systemId(): string {
    return this.configService.getOrThrow<string>('SYSTEM_ID');
  }

  /**
   * Submit an ID document for verification.
   * Returns the job ID immediately; processing happens asynchronously.
   */
  async submitDocument(
    file: Express.Multer.File,
    dto: SubmitDocumentDto,
  ): Promise<{ jobId: string; status: string; message: string }> {
    const form = new FormData();
    form.append('file', file.buffer, {
      filename: file.originalname || 'document.jpg',
      contentType: file.mimetype,
    });
    form.append('user_id', dto.userId);
    form.append('document_type', dto.documentType);
    form.append('expected_name', dto.expectedName);
    form.append('system_id', this.systemId);

    if (dto.expectedDob) form.append('expected_dob', dto.expectedDob);
    if (dto.callbackUrl) form.append('callback_url', dto.callbackUrl);

    try {
      const response = await firstValueFrom(
        this.httpService.post(`${this.baseUrl}/api/v1/verify`, form, {
          headers: {
            ...this.authHeaders,
            ...form.getHeaders(),
          },
          timeout: 30000,
        }),
      );
      const data = response.data;
      return {
        jobId: data.job_id,
        status: data.status,
        message: data.message,
      };
    } catch (error: any) {
      this.logger.error('Failed to submit document', error?.response?.data ?? error.message);
      throw new HttpException(
        error?.response?.data?.detail ?? 'Verification service unavailable',
        error?.response?.status ?? HttpStatus.BAD_GATEWAY,
      );
    }
  }

  /**
   * Poll the status of a previously submitted verification job.
   */
  async getJobStatus(jobId: string): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService.get(`${this.baseUrl}/api/v1/status/${jobId}`, {
          headers: this.authHeaders,
          timeout: 10000,
        }),
      );
      return response.data;
    } catch (error: any) {
      this.logger.error(`Failed to get status for job ${jobId}`, error?.response?.data ?? error.message);
      throw new HttpException(
        error?.response?.data?.detail ?? 'Could not retrieve job status',
        error?.response?.status ?? HttpStatus.BAD_GATEWAY,
      );
    }
  }

  /**
   * Fetch all documents pending manual review for this system.
   * Intended for admin panels.
   */
  async getReviewQueue(): Promise<{ items: any[]; total: number }> {
    try {
      const response = await firstValueFrom(
        this.httpService.get(`${this.baseUrl}/api/v1/review/queue`, {
          headers: this.authHeaders,
          params: { system_id: this.systemId },
          timeout: 10000,
        }),
      );
      return response.data;
    } catch (error: any) {
      this.logger.error('Failed to fetch review queue', error?.response?.data ?? error.message);
      throw new HttpException(
        error?.response?.data?.detail ?? 'Could not retrieve review queue',
        error?.response?.status ?? HttpStatus.BAD_GATEWAY,
      );
    }
  }

  /**
   * Approve or reject a flagged document. Call from admin panel.
   */
  async reviewDocument(jobId: string, dto: ReviewActionDto): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService.post(
          `${this.baseUrl}/api/v1/review/${jobId}`,
          {
            action: dto.action,
            comment: dto.comment,
            reviewed_by: dto.reviewedBy,
          },
          {
            headers: { ...this.authHeaders, 'Content-Type': 'application/json' },
            timeout: 10000,
          },
        ),
      );
      return response.data;
    } catch (error: any) {
      this.logger.error(`Failed to review job ${jobId}`, error?.response?.data ?? error.message);
      throw new HttpException(
        error?.response?.data?.detail ?? 'Could not submit review decision',
        error?.response?.status ?? HttpStatus.BAD_GATEWAY,
      );
    }
  }
}
