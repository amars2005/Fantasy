CREATE TABLE "leagues" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text DEFAULT 'My league' NOT NULL,
	"config" jsonb NOT NULL,
	"format" text NOT NULL,
	"board" jsonb,
	"board_hash" text,
	"adp_as_of" date,
	"frozen_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "picks" (
	"league_id" text NOT NULL,
	"seq" integer NOT NULL,
	"player_id" text,
	"taken_by" text DEFAULT 'other' NOT NULL,
	"voided_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "picks_league_id_seq_pk" PRIMARY KEY("league_id","seq")
);
--> statement-breakpoint
ALTER TABLE "picks" ADD CONSTRAINT "picks_league_id_leagues_id_fk" FOREIGN KEY ("league_id") REFERENCES "public"."leagues"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "picks_by_league" ON "picks" USING btree ("league_id","seq");