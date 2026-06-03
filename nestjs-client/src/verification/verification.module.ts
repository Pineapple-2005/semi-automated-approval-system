import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ConfigModule } from '@nestjs/config';
import { EventEmitterModule } from '@nestjs/event-emitter';

import { VerificationService } from './verification.service';
import { VerificationController } from './verification.controller';
import { VerificationAdminController } from './verification-admin.controller';
import { VerificationWebhookController } from './verification-webhook.controller';

/**
 * VerificationModule — drop this into your NestJS application's AppModule.
 *
 * Required environment variables in your .env:
 *   VERIFY_API_URL   URL of the verification microservice  (e.g. http://localhost:8000)
 *   VERIFY_API_KEY   API key issued to your system          (e.g. sk-system1-key-change-me)
 *   SYSTEM_ID        Your system's identifier               (e.g. system1)
 *
 * Usage in AppModule:
 *   @Module({ imports: [VerificationModule, ...] })
 *
 * After import, your app gains:
 *   POST   /verification                    — users submit ID documents
 *   GET    /verification/status/:jobId      — poll job status
 *   GET    /admin/verification/queue        — admin review queue
 *   POST   /admin/verification/review/:jobId — admin approve/reject
 *   POST   /webhooks/verification           — receive async results
 *
 * Listen for results anywhere in your app:
 *   @OnEvent('verification.result')
 *   handleResult(payload: VerificationWebhookPayload) { ... }
 */
@Module({
  imports: [
    // HttpModule provides Axios-based HTTP client (VerificationService uses it)
    HttpModule.register({
      timeout: 30000,
      maxRedirects: 3,
    }),
    // ConfigModule makes env vars available via ConfigService
    ConfigModule,
    // EventEmitterModule enables cross-module event bus for webhook payloads
    EventEmitterModule.forRoot(),
  ],
  controllers: [
    VerificationController,
    VerificationAdminController,
    VerificationWebhookController,
  ],
  providers: [VerificationService],
  exports: [VerificationService],
})
export class VerificationModule {}
