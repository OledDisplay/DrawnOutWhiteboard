import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';

void main() {
  runApp(const WhiteboardApp());
}

class WhiteboardApp extends StatelessWidget {
  const WhiteboardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: VectorViewerScreen(),
    );
  }
}

class VectorViewerScreen extends StatefulWidget {
  const VectorViewerScreen({super.key});

  @override
  State<VectorViewerScreen> createState() => _VectorViewerScreenState();
}

class _VectorViewerScreenState extends State<VectorViewerScreen> {
  // Hardcoded path to vectors JSON
  static const String vectorsJsonPath = r'C:\Users\marti\Code\DrawnOutWhiteboard\whiteboard_backend\StrokeVectors\edges_0.json';

  List<StrokePolyline> _polyStrokes = const [];
  List<StrokeCubic> _cubicStrokes = const [];
  String _status = 'Idle';

  Future<void> _loadAndRender() async {
    setState(() => _status = 'Loading vectors…');

    try {
      final file = File(vectorsJsonPath);
      if (!file.existsSync()) {
        setState(() => _status = 'Not found:\n$vectorsJsonPath');
        return;
      }

      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['strokes'] is! List) {
        setState(() => _status = 'Invalid JSON format (no "strokes").');
        return;
      }

      final format = (decoded['vector_format'] as String?)?.toLowerCase() ?? 'polyline';
      final List strokesJson = decoded['strokes'] as List;

      final poly = <StrokePolyline>[];
      final cubics = <StrokeCubic>[];

      if (format == 'bezier_cubic') {
        for (final s in strokesJson) {
          if (s is! Map || s['segments'] is! List) continue;
          final List segsJson = s['segments'] as List;
          final segs = <CubicSegment>[];
          for (final seg in segsJson) {
            // Expect [x0,y0, cx1,cy1, cx2,cy2, x1,y1]
            if (seg is List && seg.length >= 8) {
              final p0 = Offset((seg[0] as num).toDouble(), (seg[1] as num).toDouble());
              final c1 = Offset((seg[2] as num).toDouble(), (seg[3] as num).toDouble());
              final c2 = Offset((seg[4] as num).toDouble(), (seg[5] as num).toDouble());
              final p1 = Offset((seg[6] as num).toDouble(), (seg[7] as num).toDouble());
              segs.add(CubicSegment(p0: p0, c1: c1, c2: c2, p1: p1));
            }
          }
          if (segs.isNotEmpty) cubics.add(StrokeCubic(segs));
        }
      } else {
        // Fallback for old polyline JSONs: { points: [[x,y], ...] }
        for (final s in strokesJson) {
          if (s is! Map || s['points'] is! List) continue;
          final List pts = s['points'] as List;
          final points = <Offset>[];
          for (final p in pts) {
            if (p is List && p.length >= 2) {
              points.add(Offset((p[0] as num).toDouble(), (p[1] as num).toDouble()));
            }
          }
          if (points.length >= 2) poly.add(StrokePolyline(points));
        }
      }

      setState(() {
        _polyStrokes = poly;
        _cubicStrokes = cubics;

        final polyPts = poly.fold<int>(0, (s, e) => s + e.points.length);
        final cubicSegs = cubics.fold<int>(0, (s, e) => s + e.segments.length);
        _status = (format == 'bezier_cubic')
            ? 'Loaded cubic. Strokes: ${cubics.length}, cubic segments: $cubicSegs'
            : 'Loaded polylines. Strokes: ${poly.length}, points: $polyPts';
      });
    } catch (e, st) {
      setState(() => _status = 'Error: $e');
      // ignore: avoid_print
      print(st);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // LEFT: Whiteboard display
          Expanded(
            flex: 3,
            child: Container(
              color: Colors.white,
              child: CustomPaint(
                painter: WhiteboardPainter(_polyStrokes, _cubicStrokes),
                child: Container(),
              ),
            ),
          ),
          // RIGHT: Control panel
          Container(
            width: 280,
            color: const Color(0xFF111111),
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Control Panel',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _loadAndRender,
                  child: const Text('Load & Render vectors'),
                ),
                const SizedBox(height: 12),
                Text(
                  _status,
                  style: const TextStyle(color: Colors.white70),
                ),
                const SizedBox(height: 12),
                const Text(
                  'JSON:\nwhiteboard_backend\\StrokeVectors\\edges_0.json',
                  style: TextStyle(color: Colors.white38, fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/* ---------- Data types ---------- */

class StrokePolyline {
  final List<Offset> points;
  const StrokePolyline(this.points);
}

class CubicSegment {
  final Offset p0; // segment start (redundant across segments, but present in JSON)
  final Offset c1;
  final Offset c2;
  final Offset p1; // segment end
  const CubicSegment({required this.p0, required this.c1, required this.c2, required this.p1});
}

class StrokeCubic {
  final List<CubicSegment> segments; // consecutive cubic segments
  const StrokeCubic(this.segments);
}

/* ---------- Painter ---------- */

class WhiteboardPainter extends CustomPainter {
  final List<StrokePolyline> polylines;
  final List<StrokeCubic> cubics;
  const WhiteboardPainter(this.polylines, this.cubics);

 @override
  void paint(Canvas canvas, Size size) {
    if (polylines.isEmpty && cubics.isEmpty) return;

    final bounds = _computeBounds(polylines, cubics);
    final scale = _computeUniformScale(bounds, size, padding: 20);
    final tx = (size.width  - bounds.width  * scale) / 2 - bounds.left * scale;
    final ty = (size.height - bounds.height * scale) / 2 - bounds.top  * scale;

    canvas.save();
    canvas.translate(tx, ty);
    canvas.scale(scale);

    final paint = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      // keep ~1px in image space (don’t fatten when scaled up)
      ..strokeWidth = 1.0 / scale.clamp(1.0, double.infinity);

    // Cubics
    for (final s in cubics) {
      if (s.segments.isEmpty) continue;
      final path = Path();
      Offset? last;
      for (final seg in s.segments) {
        // ensure continuity if a segment starts away from the previous end
        if (last == null || (last! - seg.p0).distance > 1e-3) {
          path.moveTo(seg.p0.dx, seg.p0.dy);
        }
        path.cubicTo(seg.c1.dx, seg.c1.dy, seg.c2.dx, seg.c2.dy, seg.p1.dx, seg.p1.dy);
        last = seg.p1;
      }
      canvas.drawPath(path, paint);
    }

    // Polylines (fallback)
    for (final s in polylines) {
      if (s.points.length < 2) continue;
      final path = Path()..moveTo(s.points.first.dx, s.points.first.dy);
      for (int i = 1; i < s.points.length; i++) {
        path.lineTo(s.points[i].dx, s.points[i].dy);
      }
      canvas.drawPath(path, paint);
    }

    canvas.restore();
  }


  Rect _computeBounds(List<StrokePolyline> polys, List<StrokeCubic> cubics) {
    double minX = double.infinity, minY = double.infinity;
    double maxX = -double.infinity, maxY = -double.infinity;

    for (final s in polys) {
      for (final p in s.points) {
        if (p.dx < minX) minX = p.dx;
        if (p.dy < minY) minY = p.dy;
        if (p.dx > maxX) maxX = p.dx;
        if (p.dy > maxY) maxY = p.dy;
      }
    }
    for (final s in cubics) {
      for (final seg in s.segments) {
        // include all control points and end points in bounds
        for (final p in [seg.p0, seg.c1, seg.c2, seg.p1]) {
          if (p.dx < minX) minX = p.dx;
          if (p.dy < minY) minY = p.dy;
          if (p.dx > maxX) maxX = p.dx;
          if (p.dy > maxY) maxY = p.dy;
        }
      }
    }

    if (minX == double.infinity) return const Rect.fromLTWH(0, 0, 1, 1);
    // avoid zero width/height
    final w = math.max(1e-3, maxX - minX);
    final h = math.max(1e-3, maxY - minY);
    return Rect.fromLTWH(minX, minY, w, h);
  }

  double _computeUniformScale(Rect bounds, Size size, {double padding = 10}) {
    final sx = (size.width - 2 * padding) / bounds.width;
    final sy = (size.height - 2 * padding) / bounds.height;
    final v = math.min(sx, sy);
    return (v.isFinite && v > 0) ? v : 1.0;
  }

  @override
  bool shouldRepaint(covariant WhiteboardPainter old) =>
      old.polylines != polylines || old.cubics != cubics;
}
