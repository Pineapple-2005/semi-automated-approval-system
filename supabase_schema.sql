-- =============================================================================
-- Philippine ID Document Verification API - Supabase Schema
-- =============================================================================
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New query).
-- Safe to run multiple times: uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.

-- Enable the pgcrypto extension if not already enabled (required for gen_random_uuid()).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Main table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS verification_jobs (

    -- Primary key
    job_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Tenant / user identity
    system_id           VARCHAR(100)    NOT NULL,
    user_id             VARCHAR(100)    NOT NULL,

    -- Document metadata
    document_type       VARCHAR(50)     NOT NULL,
    expected_name       VARCHAR(200)    NOT NULL,
    expected_dob        VARCHAR(20),
    callback_url        VARCHAR(500),

    -- Processing state
    -- status values: processing | needs_review | approved | rejected | processing_error
    status              VARCHAR(20)     NOT NULL DEFAULT 'processing',

    -- Concerns surfaced by OCR pipeline (reasons reviewer should look closely)
    flags               JSONB           NOT NULL DEFAULT '[]',

    -- Confirmations surfaced by OCR pipeline (reasons reviewer may approve)
    approval_signals    JSONB           NOT NULL DEFAULT '[]',

    -- OCR results
    extracted_data      JSONB           NOT NULL DEFAULT '{}',
    confidence_scores   JSONB           NOT NULL DEFAULT '{}',

    -- Classifier output
    document_type_detected  VARCHAR(50),
    image_path              VARCHAR(500) NOT NULL,

    -- Timestamps
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Human review audit trail
    reviewed_by         VARCHAR(100),
    review_comment      TEXT,
    reviewed_at         TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS ix_verification_jobs_system_id
    ON verification_jobs (system_id);

CREATE INDEX IF NOT EXISTS ix_verification_jobs_status
    ON verification_jobs (status);

CREATE INDEX IF NOT EXISTS ix_verification_jobs_created_at
    ON verification_jobs (created_at DESC);

-- ---------------------------------------------------------------------------
-- Migration: add approval_signals if upgrading from the old auto-approval schema
-- ---------------------------------------------------------------------------

ALTER TABLE verification_jobs
    ADD COLUMN IF NOT EXISTS approval_signals JSONB NOT NULL DEFAULT '[]';
