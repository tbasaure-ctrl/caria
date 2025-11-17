# Quick Deployment Guide

## Deploy Frontend Changes
```bash
git add .
git commit -m "Your changes"
git push origin main
```
Vercel auto-deploys in ~2 minutes.

## Deploy Backend Changes
```bash
git add .
git commit -m "Your changes"
git push origin main
```
Cloud Build auto-deploys in ~5-8 minutes.

## Check Deployment Status
```bash
# Backend
gcloud builds list --limit=1

# Check if live
curl https://caria-api-418525923468.us-central1.run.app/health
```

## What We Just Added
✅ Resources section (lectures/articles)
✅ Reddit hot stocks widget
✅ Auto CORS for all Vercel URLs

## Your URLs
- Frontend: Check https://vercel.com/dashboard
- Backend: https://caria-api-418525923468.us-central1.run.app
