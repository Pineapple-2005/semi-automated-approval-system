import { Controller, Post, Body, Logger, HttpCode, HttpStatus } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';

export interface VerificationWebhookPayload {
  job_id: string;
  status: 'auto_approved' | 'approved' | 'rejected';
  flags: string[];
  extracted_data: Record<string, any>;
  system_id: string;
  document_type: string;
  timestamp: string;
}

/**
 * Receives asynchronous webhook callbacks from the Verification API.
 *
 * Point your VERIFY_API_URL-backed server's callback_url to:
 *   POST https://your-app.com/webhooks/verification
 *
 * The verification API POSTs here after a job reaches its final status
 * (auto_approved, approved, or rejected).
 *
 * This controller emits a "verification.result" event via EventEmitter2 so
 * your business logic can react without coupling to this controller.
 *
 * TODO: Add HMAC signature validation here if you add a shared webhook secret
 * to your verification API deployment.
 */
@Controller('webhooks/verification')
export class VerificationWebhookController {
  private readonly logger = new Logger(VerificationWebhookController.name);

  constructor(private readonly eventEmitter: EventEmitter2) {}

  @Post()
  @HttpCode(HttpStatus.OK)
  receiveWebhook(@Body() payload: VerificationWebhookPayload): { received: boolean } {
    this.logger.log(
      `Webhook received: job=${payload.job_id} status=${payload.status} flags=${payload.flags?.length ?? 0}`,
    );

    // Emit event so other parts of the app can subscribe via @OnEvent('verification.result')
    this.eventEmitter.emit('verification.result', payload);

    // TODO: Optionally persist webhook payload to your own database here, e.g.:
    //   await this.yourApplicationService.handleVerificationResult(payload);

    return { received: true };
  }
}
