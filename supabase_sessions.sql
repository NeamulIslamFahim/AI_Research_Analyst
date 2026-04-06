create table if not exists public.app_sessions (
    owner_id text not null,
    session_id text not null,
    title text,
    mode text,
    updated_at timestamptz not null default timezone('utc', now()),
    payload jsonb not null,
    primary key (owner_id, session_id)
);

create index if not exists app_sessions_owner_updated_idx
    on public.app_sessions (owner_id, updated_at desc);

