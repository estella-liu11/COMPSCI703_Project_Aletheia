-- =====================================================================
-- Aletheia — audits table + Row Level Security
-- =====================================================================
-- Run this once in Supabase Dashboard → SQL Editor → New query.
--
-- What it creates:
--   1. Table `audits` — one row per completed Chief-Officer report.
--   2. RLS policies — every user can only read / write their OWN rows.
--   3. An index on (user_id, created_at) for fast "newest-first" listing.
-- =====================================================================

-- --- 1. Table ---------------------------------------------------------
create table if not exists public.audits (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid not null default auth.uid()
                       references auth.users (id) on delete cascade,
    file_name     text not null,
    final_report  text not null,
    created_at    timestamptz not null default now()
);

comment on table public.audits is
  'One row per Aletheia Chief-Officer final report. user_id is auto-filled by RLS.';

-- --- 2. Index ---------------------------------------------------------
create index if not exists audits_user_created_idx
    on public.audits (user_id, created_at desc);

-- --- 3. Row Level Security -------------------------------------------
alter table public.audits enable row level security;

-- Drop existing policies if re-running this script
drop policy if exists "audits_select_own" on public.audits;
drop policy if exists "audits_insert_own" on public.audits;
drop policy if exists "audits_delete_own" on public.audits;

-- Each user sees only their own rows
create policy "audits_select_own"
    on public.audits
    for select
    using (user_id = auth.uid());

-- Each user can insert rows only for themselves
create policy "audits_insert_own"
    on public.audits
    for insert
    with check (user_id = auth.uid());

-- (Optional) allow users to delete their own audits
create policy "audits_delete_own"
    on public.audits
    for delete
    using (user_id = auth.uid());

-- =====================================================================
-- Sanity check: after running this, the SQL Editor should show
--   "Success. No rows returned" and Table Editor should list `audits`
--   under public schema with RLS enabled (lock icon).
-- =====================================================================
