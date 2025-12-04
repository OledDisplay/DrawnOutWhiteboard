import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;  // <-- ADD THIS


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


class _VectorViewerScreenState extends State<VectorViewerScreen>
    with SingleTickerProviderStateMixin {
  // === BASE FOLDER FOR JSONS (IMAGES) ===
  // === BASE FOLDER FOR JSONS (IMAGES) ===
  static String get _vectorsFolder =>
      _resolveBackendSubdir('StrokeVectors');

  // FONT GLYPH JSONS (TEXT)
  static String get _fontVectorsFolder =>
      _resolveBackendSubdir('Font');

  static String get _fontMetricsPath =>
      '${_fontVectorsFolder}\\font_metrics.json';

  // RAW strokes from last loaded file (kept for debugging)
  List<StrokePolyline> _polyStrokes = const [];
  List<StrokeCubic> _cubicStrokes = const [];

  // DRAWABLE strokes on the board
  List<DrawableStroke> _drawableStrokes = const [];
  List<DrawableStroke> _staticStrokes = const [];
  List<DrawableStroke> _animStrokes = const [];

  double? _srcWidth;
  double? _srcHeight;

  String _status = 'Idle';

  late final AnimationController _controller;
  double _animValue = 0.0;

  // step mode
  bool _stepMode = false;
  int _stepStrokeCount = 0;

  static const double _targetResolution = 2000.0; // target max side
  static const double _basePenWidthPx = 3.0; // logical image px

  // ------------ Backend API config ------------
  static const String _apiBaseUrl = 'http://127.0.0.1:8000'; // change if needed
  static const bool _backendEnabled = true;

  Uri _apiUri(String path) => Uri.parse('$_apiBaseUrl$path');


  // Virtual board extents in world coordinates.
  static const double _boardWidth = _targetResolution;
  static const double _boardHeight = _targetResolution;

  // Max number of points we keep per stroke for display.
  static const int _maxDisplayPointsPerStroke = 120;

  // ------------------- CORE TIMING PARAMETERS (NON-TEXT) -------------------

  // Stroke draw timing (seconds) for normal objects/images
  double _minStrokeTimeSec = 0.18;
  double _maxStrokeTimeSec = 0.32;

  // Extra time from length: seconds per 1000px of stroke length
  double _lengthTimePerKPxSec = 0.08;

  // Extra time from curvature: max seconds added at "full" curvature
  double _curvatureExtraMaxSec = 0.08;

  // Curvature profile along the stroke (local slowdowns)
  double _curvatureProfileFactor = 1.5;
  double _curvatureAngleScale = 80.0;

  // Travel / pause between strokes (seconds) for normal objects
  double _baseTravelTimeSec = 0.15;
  double _travelTimePerKPxSec = 0.12;
  double _minTravelTimeSec = 0.15;
  double _maxTravelTimeSec = 0.35;

  // Global animation timing
  double _globalSpeedMultiplier = 1.0;

  double _textLetterGapPx = 20.0; // default gap in board pixels between letters


  // --------------- MULTI-OBJECT SUPPORT ---------------

  // UI controllers for loading JSONs (images)
  final TextEditingController _fileNameController =
      TextEditingController(text: 'edges_0_skeleton.json');
  final TextEditingController _posXController =
      TextEditingController(text: '0');
  final TextEditingController _posYController =
      TextEditingController(text: '0');
  final TextEditingController _scaleController =
      TextEditingController(text: '1.0');

  // list of distinct json names currently on board (including text prompts)
  final List<String> _drawnJsonNames = [];
  String? _selectedEraseName;

  // --------------- TEXT ENGINE STATE ---------------

  // cached glyphs by code unit
  final Map<int, GlyphData> _glyphCache = {};
  double? _fontLineHeightPx; // ascent+descent in font pixels
  double? _fontImageHeightPx; // image_height used in font generation

  // text timing: when _animIsText = true, we use these rules
  bool _animIsText = false;
  double _textStrokeBaseTimeSec = 0.035; // constant per stroke
  double _textStrokeCurveExtraFrac = 0.25; // fraction of base added at max curvature
  double _textLetterPauseSec = 0.0; // currently unused (no waits)

  // reference used only for text scaling UI defaults
  double _textBaseFontSizeRef = 200.0;

  // text UI controllers
  final TextEditingController _textPromptController =
      TextEditingController(text: 'Hello, world');
  final TextEditingController _textXController =
      TextEditingController(text: '0');
  final TextEditingController _textYController =
      TextEditingController(text: '0');
  final TextEditingController _textSizeController =
      TextEditingController(text: '180');
  final TextEditingController _textGapController =
    TextEditingController(text: '20'); 

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 20000),
    )
      ..addListener(() {
        if (!_stepMode) {
          setState(() {
            _animValue = _controller.value;
          });
        }
      })
      ..addStatusListener((status) {
        if (status == AnimationStatus.completed) {
          if (_animStrokes.isNotEmpty) {
            setState(() {
              _staticStrokes = [..._staticStrokes, ..._animStrokes];
              _animStrokes = const [];
              _drawableStrokes = [..._staticStrokes];
              _animValue = 1.0;
              _status =
                  'Animation finished. Total strokes: ${_drawableStrokes.length}';
            });
          }
        }
      });

    // Load any objects saved on the backend (if enabled)
    _loadObjectsFromBackend();
  }


  

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // ------------------- FONT METRICS (TEXT) -------------------

  Future<void> _ensureFontMetricsLoaded() async {
    if (_fontLineHeightPx != null && _fontImageHeightPx != null) return;
    try {
      final file = File(_fontMetricsPath);
      if (!file.existsSync()) {
        _fontImageHeightPx = _targetResolution;
        _fontLineHeightPx = _targetResolution * 0.5;
        return;
      }
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is Map) {
        final lh = (decoded['line_height_px'] as num?)?.toDouble();
        final ih = (decoded['image_height'] as num?)?.toDouble();
        if (lh != null && lh > 0) _fontLineHeightPx = lh;
        if (ih != null && ih > 0) _fontImageHeightPx = ih;
      }
      _fontLineHeightPx ??= _targetResolution * 0.5;
      _fontImageHeightPx ??= _targetResolution;
    } catch (_) {
      _fontLineHeightPx = _targetResolution * 0.5;
      _fontImageHeightPx = _targetResolution;
    }
  }

    // ========== BACKEND SYNC HELPERS ==========

  Future<void> _syncCreateImageOnBackend({
    required String fileName,
    required Offset origin,
    required double scale,
  }) async {
    if (!_backendEnabled) return;
    final uri = _apiUri('/api/whiteboard/objects/image/');
    final body = json.encode({
      'file_name': fileName,
      'x': origin.dx,
      'y': origin.dy,
      'scale': scale,
    });

    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );
    if (resp.statusCode >= 400) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  Future<void> _syncCreateTextOnBackend({
    required String prompt,
    required Offset origin,
    required double letterSize,
    required double letterGap,
  }) async {
    if (!_backendEnabled) return;
    final uri = _apiUri('/api/whiteboard/objects/text/');
    final body = json.encode({
      'prompt': prompt,
      'x': origin.dx,
      'y': origin.dy,
      'letter_size': letterSize,
      'letter_gap': letterGap,
    });

    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );
    if (resp.statusCode >= 400) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  Future<void> _syncDeleteOnBackend(String name) async {
    if (!_backendEnabled) return;
    final uri = _apiUri('/api/whiteboard/objects/delete/');
    final body = json.encode({'name': name});

    final resp = await http.delete(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );

    // 404 is ok – already deleted backend-side
    if (resp.statusCode >= 400 && resp.statusCode != 404) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  Future<void> _loadObjectsFromBackend() async {
    if (!_backendEnabled) return;
    try {
      final uri = _apiUri('/api/whiteboard/objects/');
      final resp = await http.get(uri);
      if (resp.statusCode != 200) {
        setState(() {
          _status = 'Backend list error: HTTP ${resp.statusCode}';
        });
        return;
      }

      final decoded = json.decode(resp.body);
      if (decoded is! Map || decoded['objects'] is! List) {
        setState(() {
          _status = 'Backend list invalid format';
        });
        return;
      }

      final List objs = decoded['objects'] as List;

      for (final o in objs) {
        if (o is! Map) continue;
        final name = (o['name'] ?? '').toString();
        final kind = (o['kind'] ?? '').toString();
        final double x = (o['pos_x'] as num?)?.toDouble() ?? 0.0;
        final double y = (o['pos_y'] as num?)?.toDouble() ?? 0.0;
        final double scale =
            (o['scale'] as num?)?.toDouble() ?? 1.0;

        if (kind == 'image') {
          // pure local draw, no backend call
          await _addObjectFromJsonInternal(
            fileName: name,
            origin: Offset(x, y),
            objectScale: scale,
          );
        } else if (kind == 'text') {
          final double letterSize =
              (o['letter_size'] as num?)?.toDouble() ?? _textBaseFontSizeRef;
          final double letterGap =
              (o['letter_gap'] as num?)?.toDouble() ?? _textLetterGapPx;

          // use stored gap for this text batch
          _textLetterGapPx = letterGap;
          await _writeTextPromptLocal(
            prompt: name,
            origin: Offset(x, y),
            letterSize: letterSize,
          );
        }
      }

      // After initial load, flatten any remaining anim strokes into static
      if (mounted) {
        _controller.stop();
        setState(() {
          _staticStrokes = [..._staticStrokes, ..._animStrokes];
          _animStrokes = const [];
          _drawableStrokes = [..._staticStrokes];
          _animValue = 1.0;
          _status = 'Synced ${objs.length} object(s) from backend.';
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Backend sync failed: $e';
      });
    }
  }


  // ------------------- LOADING & BUILDING (IMAGES) -------------------

  Future<void> _loadAndRender() async {
    final fileName = _fileNameController.text.trim();
    if (fileName.isEmpty) {
      setState(() {
        _status = 'No file name specified.';
      });
      return;
    }

    final x = double.tryParse(_posXController.text.trim()) ?? 0.0;
    final y = double.tryParse(_posYController.text.trim()) ?? 0.0;
    final scale = double.tryParse(_scaleController.text.trim()) ?? 1.0;

    await _addObjectFromJson(
      fileName: fileName,
      origin: Offset(x, y),
      objectScale: scale,
    );
  }

    // Public entry: UI + external callers
  Future<void> _addObjectFromJson({
    required String fileName,
    required Offset origin,
    required double objectScale,
  }) async {
    // Local drawing first (existing behavior)
    await _addObjectFromJsonInternal(
      fileName: fileName,
      origin: origin,
      objectScale: objectScale,
    );

    // Fire-and-forget backend sync
    if (_backendEnabled) {
      () async {
        try {
          await _syncCreateImageOnBackend(
            fileName: fileName,
            origin: origin,
            scale: objectScale,
          );
        } catch (e) {
          if (!mounted) return;
          setState(() {
            _status += '\n[Backend create image failed: $e]';
          });
        }
      }();
    }
  }

  // Internal: your previous local logic moved here, unchanged
  Future<void> _addObjectFromJsonInternal({
    required String fileName,
    required Offset origin,
    required double objectScale,
  }) async {
    final path = '$_vectorsFolder\\$fileName';
    setState(() => _status = 'Loading $fileName…');

    final file = File(path);
    if (!file.existsSync()) {
      setState(() => _status = 'Not found:\n$path');
      return;
    }

    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['strokes'] is! List) {
        setState(() => _status = 'Invalid JSON format (no "strokes").');
        return;
      }

      final format =
          (decoded['vector_format'] as String?)?.toLowerCase() ?? 'polyline';
      final List strokesJson = decoded['strokes'] as List;

      final poly = <StrokePolyline>[];
      final cubics = <StrokeCubic>[];

      final srcWidth = (decoded['width'] as num?)?.toDouble();
      final srcHeight = (decoded['height'] as num?)?.toDouble();
      _srcWidth = srcWidth;
      _srcHeight = srcHeight;

      if (format == 'bezier_cubic') {
        for (final s in strokesJson) {
          if (s is! Map || s['segments'] is! List) continue;
          final List segsJson = s['segments'] as List;
          final segs = <CubicSegment>[];
          for (final seg in segsJson) {
            if (seg is List && seg.length >= 8) {
              final p0 = Offset(
                  (seg[0] as num).toDouble(), (seg[1] as num).toDouble());
              final c1 = Offset(
                  (seg[2] as num).toDouble(), (seg[3] as num).toDouble());
              final c2 = Offset(
                  (seg[4] as num).toDouble(), (seg[5] as num).toDouble());
              final p1 = Offset(
                  (seg[6] as num).toDouble(), (seg[7] as num).toDouble());
              segs.add(CubicSegment(p0: p0, c1: c1, c2: c2, p1: p1));
            }
          }
          if (segs.isNotEmpty) cubics.add(StrokeCubic(segs));
        }
      } else {
        for (final s in strokesJson) {
          if (s is! Map || s['points'] is! List) continue;
          final List pts = s['points'] as List;
          final points = <Offset>[];
          for (final p in pts) {
            if (p is List && p.length >= 2) {
              points.add(
                  Offset((p[0] as num).toDouble(), (p[1] as num).toDouble()));
            }
          }
          if (points.length >= 2) poly.add(StrokePolyline(points));
        }
      }

      _polyStrokes = poly;
      _cubicStrokes = cubics;

      double useWidth = srcWidth ?? 1000.0;
      double useHeight = srcHeight ?? 1000.0;
      if ((srcWidth == null || srcHeight == null) &&
          (poly.isNotEmpty || cubics.isNotEmpty)) {
        final bounds = _computeRawBounds(poly, cubics);
        useWidth = bounds.width;
        useHeight = bounds.height;
        _srcWidth = useWidth;
        _srcHeight = useHeight;
      }

      final newStrokes = _buildDrawableStrokesForObject(
        jsonName: fileName,
        origin: origin,
        objectScale: objectScale,
        polylines: poly,
        cubics: cubics,
        srcWidth: useWidth,
        srcHeight: useHeight,
        targetResolution: _targetResolution,
        basePenWidth: _basePenWidthPx,
      );

      setState(() {
        if (_animStrokes.isNotEmpty) {
          _controller.stop();
          _staticStrokes = [..._staticStrokes, ..._animStrokes];
          _animStrokes = const [];
          _animValue = 0.0;
        }

        _animIsText = false; // this batch is not text

        _animStrokes = newStrokes;
        _drawableStrokes = [..._staticStrokes, ..._animStrokes];

        if (!_drawnJsonNames.contains(fileName)) {
          _drawnJsonNames.add(fileName);
        }
        _selectedEraseName ??= fileName;

        final polyPts = poly.fold<int>(0, (s, e) => s + e.points.length);
        final cubicSegs = cubics.fold<int>(0, (s, e) => s + e.segments.length);
        _status =
            'Added $fileName\nFormat: $format | strokes: poly=${poly.length}, cubic=${cubics.length}, pts=$polyPts, segs=$cubicSegs\nTotal drawable strokes: ${_drawableStrokes.length}';
      });

      _recomputeTimingForAnimStrokes();
    } catch (e, st) {
      setState(() => _status = 'Error loading $fileName: $e');
      // ignore: avoid_print
      print(st);
    }
  }


  // ------------------- GLYPH LOADING (TEXT) -------------------

  /// Find the whiteboard_backend folder starting from Directory.current
  /// and walking upwards. Then append [subdir].
  static String _resolveBackendSubdir(String subdir) {
    // Start where the process is running (for Windows desktop:
    // build/windows/x64/runner/Debug by default).
    var dir = Directory.current;

    // Walk up at most 10 levels to be safe.
    for (int i = 0; i < 10; i++) {
      final candidate =
          Directory('${dir.path}\\whiteboard_backend');
      if (candidate.existsSync()) {
        if (subdir.isEmpty) return candidate.path;
        return '${candidate.path}\\$subdir';
      }

      final parent = dir.parent;
      if (parent.path == dir.path) {
        // Reached filesystem root; stop.
        break;
      }
      dir = parent;
    }

    // Fallback: just assume current dir has whiteboard_backend next to it.
    // This will fail loudly if it's wrong.
    return '${Directory.current.path}\\whiteboard_backend\\$subdir';
  }


  Future<GlyphData?> _getGlyphForCode(int codeUnit) async {
    if (_glyphCache.containsKey(codeUnit)) {
      return _glyphCache[codeUnit];
    }

    final hex = codeUnit.toRadixString(16).padLeft(4, '0');
    final path = '$_fontVectorsFolder\\$hex.json';
    final file = File(path);
    if (!file.existsSync()) {
      return null;
    }

    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['strokes'] is! List) {
        return null;
      }

      final List strokesJson = decoded['strokes'] as List;
      final format =
          (decoded['vector_format'] as String?)?.toLowerCase() ?? 'bezier_cubic';

      final cubics = <StrokeCubic>[];

      if (format == 'bezier_cubic') {
        for (final s in strokesJson) {
          if (s is! Map || s['segments'] is! List) continue;
          final List segsJson = s['segments'] as List;
          final segs = <CubicSegment>[];
          for (final seg in segsJson) {
            if (seg is List && seg.length >= 8) {
              final p0 = Offset(
                  (seg[0] as num).toDouble(), (seg[1] as num).toDouble());
              final c1 = Offset(
                  (seg[2] as num).toDouble(), (seg[3] as num).toDouble());
              final c2 = Offset(
                  (seg[4] as num).toDouble(), (seg[5] as num).toDouble());
              final p1 = Offset(
                  (seg[6] as num).toDouble(), (seg[7] as num).toDouble());
              segs.add(CubicSegment(p0: p0, c1: c1, c2: c2, p1: p1));
            }
          }
          if (segs.isNotEmpty) cubics.add(StrokeCubic(segs));
        }
      }

      if (cubics.isEmpty) return null;

      final bounds = _computeRawBounds(const [], cubics);
      final glyph = GlyphData(cubics: cubics, bounds: bounds);
      _glyphCache[codeUnit] = glyph;
      return glyph;
    } catch (_) {
      return null;
    }
  }

  // ------------------- TIMING RECOMPUTE -------------------

  void _recomputeTimingForAnimStrokes() {
    if (_animStrokes.isEmpty) return;

    if (_animIsText) {
      // TEXT: constant per-stroke time, no waits between strokes.
      for (final s in _animStrokes) {
        final curvature = s.curvatureMetricDeg;
        final curvNorm = (curvature / 70.0).clamp(0.0, 1.0);
        final base = _textStrokeBaseTimeSec;
        final extra = base * _textStrokeCurveExtraFrac * curvNorm;
        s.drawTimeSec = base + extra;
        s.travelTimeBeforeSec = 0.0; // no travel/pause between strokes
        s.timeWeight = s.drawTimeSec;
      }
    } else {
      // NORMAL OBJECT: original length/curvature based timing.
      for (final s in _animStrokes) {
        final length = s.lengthPx;
        final curvature = s.curvatureMetricDeg;

        final lengthK = length / 1000.0;
        final curvNorm = (curvature / 70.0).clamp(0.0, 1.0);

        final rawTime = _minStrokeTimeSec +
            lengthK * _lengthTimePerKPxSec +
            curvNorm * _curvatureExtraMaxSec;

        s.drawTimeSec =
            rawTime.clamp(_minStrokeTimeSec, _maxStrokeTimeSec).toDouble();
      }

      DrawableStroke? prev;
      for (final s in _animStrokes) {
        double travel = 0.0;

        if (prev != null) {
          final lastP = prev.points.last;
          final firstP = s.points.first;
          final dist = (firstP - lastP).distance;
          final distK = dist / 1000.0;

          final rawTravel =
              _baseTravelTimeSec + distK * _travelTimePerKPxSec;

          travel = rawTravel
              .clamp(_minTravelTimeSec, _maxTravelTimeSec)
              .toDouble();
        } else {
          travel = 0.0;
        }

        s.travelTimeBeforeSec = travel;
        s.timeWeight = s.travelTimeBeforeSec + s.drawTimeSec;

        prev = s;
      }
    }

    final totalSeconds =
        _animStrokes.fold<double>(0.0, (sum, d) => sum + d.timeWeight);

    final animSeconds =
        (totalSeconds > 0) ? (totalSeconds / _globalSpeedMultiplier) : 0.0;

    if (animSeconds <= 0.0) {
      _controller.stop();
      setState(() {
        _animValue = 1.0;
        _drawableStrokes = [..._staticStrokes, ..._animStrokes];
        _status =
            'Total strokes: ${_drawableStrokes.length} | nothing to animate';
      });
      return;
    }

    final ms = math.max(1, (animSeconds * 1000).round());
    _controller.duration = Duration(milliseconds: ms);

    _controller
      ..reset()
      ..forward();

    setState(() {
      _animValue = 0.0;
      _stepMode = false;
      _stepStrokeCount = 0;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];
      _status =
          'Total strokes: ${_drawableStrokes.length} | anim=${ms}ms (new object)';
    });
  }

  void _rebuildDrawableFromCurrentStrokes() {
    if (_animStrokes.isEmpty) return;
    _recomputeTimingForAnimStrokes();
  }

  void _restartAnimationMode() {
    if (_animStrokes.isEmpty) {
      setState(() {
        _status = 'No active animation – add an object to animate.';
      });
      return;
    }
    setState(() {
      _stepMode = false;
      _animValue = 0.0;
    });
    _controller
      ..reset()
      ..forward();
  }

  void _stepNextStroke() {
    if (_drawableStrokes.isEmpty) return;
    setState(() {
      _stepMode = true;
      _controller.stop();
      if (_stepStrokeCount < _drawableStrokes.length) {
        _stepStrokeCount += 1;
      }
    });
  }

   void _eraseObjectByName(String name) {
    if (name.isEmpty) return;
    if (_drawableStrokes.isEmpty) return;

    setState(() {
      _staticStrokes =
          _staticStrokes.where((s) => s.jsonName != name).toList();
      _animStrokes = _animStrokes.where((s) => s.jsonName != name).toList();

      _drawableStrokes = [..._staticStrokes, ..._animStrokes];

      _drawnJsonNames.remove(name);

      if (_drawnJsonNames.isEmpty) {
        _selectedEraseName = null;
      } else if (_selectedEraseName == name) {
        _selectedEraseName = _drawnJsonNames.last;
      }

      _status = 'Erased $name. Remaining strokes: ${_drawableStrokes.length}';
    });

    if (_animStrokes.isEmpty) {
      _controller.stop();
      setState(() {
        _animValue = 0.0;
        _stepMode = false;
        _stepStrokeCount = 0;
        if (_drawableStrokes.isEmpty) {
          _status = 'Board empty.';
        }
      });
    }

    // Backend delete (fire-and-forget)
    if (_backendEnabled) {
      () async {
        try {
          await _syncDeleteOnBackend(name);
        } catch (e) {
          if (!mounted) return;
          setState(() {
            _status += '\n[Backend delete failed: $e]';
          });
        }
      }();
    }
  }


  // ------------------- TEXT WRITING -------------------

    Future<void> _writeTextFromUi() async {
    final prompt = _textPromptController.text;
    if (prompt.isEmpty) {
      setState(() {
        _status = 'Text prompt is empty.';
      });
      return;
    }

    final x = double.tryParse(_textXController.text.trim()) ?? 0.0;
    final y = double.tryParse(_textYController.text.trim()) ?? 0.0;
    final size =
        double.tryParse(_textSizeController.text.trim()) ?? _textBaseFontSizeRef;

    // NEW: update gap factor from UI
    final gapParsed =
     double.tryParse(_textGapController.text.trim()) ?? _textLetterGapPx;
    _textLetterGapPx = math.max(0.0, gapParsed);

    await _writeTextPrompt(
      prompt: prompt,
      origin: Offset(x, y),
      letterSize: size,
    );
  }


   // Public entry: UI and others
  Future<void> _writeTextPrompt({
    required String prompt,
    required Offset origin,
    required double letterSize,
  }) async {
    await _writeTextPromptLocal(
      prompt: prompt,
      origin: origin,
      letterSize: letterSize,
    );

    if (_backendEnabled) {
      () async {
        try {
          await _syncCreateTextOnBackend(
            prompt: prompt,
            origin: origin,
            letterSize: letterSize,
            letterGap: _textLetterGapPx,
          );
        } catch (e) {
          if (!mounted) return;
          setState(() {
            _status += '\n[Backend text sync failed: $e]';
          });
        }
      }();
    }
  }

  // Internal: your previous local draw logic moved here
  Future<void> _writeTextPromptLocal({
    required String prompt,
    required Offset origin,
    required double letterSize,
  }) async {
    if (prompt.isEmpty) {
      setState(() {
        _status = 'Text prompt is empty.';
      });
      return;
    }

    if (letterSize <= 0) {
      letterSize = _textBaseFontSizeRef;
    }

    await _ensureFontMetricsLoaded();
    final lineHeight = _fontLineHeightPx ?? _targetResolution * 0.5;
    final imageHeight = _fontImageHeightPx ?? _targetResolution;
    final scale = letterSize / lineHeight;

    // finalize any current animation into static
    if (_animStrokes.isNotEmpty) {
      _controller.stop();
      _staticStrokes = [..._staticStrokes, ..._animStrokes];
      _animStrokes = const [];
      _animValue = 0.0;
    }

    final newStrokes = <DrawableStroke>[];
    final diagBoard =
        math.sqrt(_boardWidth * _boardWidth + _boardHeight * _boardHeight);

    double cursorX = origin.dx;
    final double baselineWorldY = origin.dy;
    final double baselineGlyph = imageHeight / 2.0;
    final double baselineGlyphScaled = baselineGlyph * scale;

    final double letterGapPx = _textLetterGapPx;
    const double spaceWidthFactor = 0.5;

    for (int i = 0; i < prompt.length; i++) {
      final ch = prompt[i];
      final code = ch.codeUnitAt(0);

      if (ch == ' ') {
        cursorX += letterSize * spaceWidthFactor;
        continue;
      }

      final glyph = await _getGlyphForCode(code);
      if (glyph == null) {
        cursorX += letterSize * spaceWidthFactor;
        continue;
      }

      final gb = glyph.bounds;
      final glyphWidth = math.max(gb.width, 1e-3);
      final glyphLeft = gb.left;

      // X anchor: place leftmost stroke of glyph at cursorX
      final double letterOffsetX = cursorX - glyphLeft * scale;

      // Y anchor: baseline alignment
      final double letterOffsetY =
          baselineWorldY - baselineGlyphScaled;

      for (final stroke in glyph.cubics) {
        final ptsRaw = _sampleCubicStroke(stroke, upscale: scale);
        if (ptsRaw.length < 2) continue;

        final ptsPlaced = ptsRaw
            .map(
              (p) => Offset(
                p.dx + letterOffsetX,
                p.dy + letterOffsetY,
              ),
            )
            .toList(growable: false);

        newStrokes.add(
          _makeDrawableFromPoints(
            jsonName: prompt, // group id for erase = whole prompt
            objectOrigin: origin,
            objectScale: scale,
            pts: ptsPlaced,
            basePenWidth: _basePenWidthPx,
            diag: diagBoard,
          ),
        );
      }

      final glyphWidthScaled = glyphWidth * scale;
      cursorX += glyphWidthScaled + letterGapPx;
    }

    if (newStrokes.isEmpty) {
      setState(() {
        _status = 'No strokes generated for text "$prompt".';
      });
      return;
    }

    setState(() {
      _animIsText = true;
      _animStrokes = newStrokes;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];

      if (!_drawnJsonNames.contains(prompt)) {
        _drawnJsonNames.add(prompt);
      }
      _selectedEraseName ??= prompt;

      _status =
          'Writing text "$prompt" | strokes=${newStrokes.length}, letters=${prompt.length}';
    });

    _recomputeTimingForAnimStrokes();
  }


  // ------------------- UI -------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // LEFT: drawing surface
          Expanded(
            flex: 3,
            child: Container(
              color: Colors.white,
              child: CustomPaint(
                painter: WhiteboardPainter(
                  staticStrokes: _staticStrokes,
                  animStrokes: _animStrokes,
                  animationT: _animValue,
                  basePenWidth: _basePenWidthPx,
                  stepMode: _stepMode,
                  stepStrokeCount: _stepStrokeCount,
                  boardWidth: _boardWidth,
                  boardHeight: _boardHeight,
                ),
              ),
            ),
          ),
          // RIGHT: control panel
          Container(
            width: 340,
            color: const Color(0xFF111111),
            padding: const EdgeInsets.all(16),
            child: SingleChildScrollView(
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
                    child: const Text('Load & Render (auto anim)'),
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: _restartAnimationMode,
                    child: const Text('Replay animation'),
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: _stepNextStroke,
                    child: const Text('Step: next stroke'),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _status,
                    style: const TextStyle(color: Colors.white70),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _stepMode
                        ? 'Mode: STEP  | shown: $_stepStrokeCount'
                        : 'Mode: ANIM | t=${_animValue.toStringAsFixed(2)}',
                    style:
                        const TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Legacy JSON path:\nwhiteboard_backend\\StrokeVectors\\edges_0_skeleton.json',
                    style: TextStyle(color: Colors.white38, fontSize: 11),
                  ),

                  // -------- JSON / PLACEMENT UI --------
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Load JSON from folder',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Folder:\n$_vectorsFolder',
                    style:
                        const TextStyle(color: Colors.white38, fontSize: 10),
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _fileNameController,
                    style:
                        const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'File name (e.g. edges_0_skeleton.json)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _posXController,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 12),
                          decoration: const InputDecoration(
                            labelText: 'X',
                            labelStyle: TextStyle(
                                color: Colors.white54, fontSize: 11),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                                horizontal: 8, vertical: 6),
                          ),
                          keyboardType:
                              const TextInputType.numberWithOptions(
                                  decimal: true, signed: true),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: TextField(
                          controller: _posYController,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 12),
                          decoration: const InputDecoration(
                            labelText: 'Y',
                            labelStyle: TextStyle(
                                color: Colors.white54, fontSize: 11),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                                horizontal: 8, vertical: 6),
                          ),
                          keyboardType:
                              const TextInputType.numberWithOptions(
                                  decimal: true, signed: true),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _scaleController,
                    style:
                        const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Scale (1.0 = default 2k size)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                        decimal: true, signed: false),
                  ),

                  // -------- TEXT WRITING UI --------
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Text writing (Font folder)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    r'Font JSON path: whiteboard_backend\Font\<hex>.json',
                    style: TextStyle(color: Colors.white38, fontSize: 10),
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _textPromptController,
                    style:
                        const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Text to write',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _textXController,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 12),
                          decoration: const InputDecoration(
                            labelText: 'Text X (baseline)',
                            labelStyle: TextStyle(
                                color: Colors.white54, fontSize: 11),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                                horizontal: 8, vertical: 6),
                          ),
                          keyboardType:
                              const TextInputType.numberWithOptions(
                                  decimal: true, signed: true),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: TextField(
                          controller: _textYController,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 12),
                          decoration: const InputDecoration(
                            labelText: 'Text Y (baseline)',
                            labelStyle: TextStyle(
                                color: Colors.white54, fontSize: 11),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                                horizontal: 8, vertical: 6),
                          ),
                          keyboardType:
                              const TextInputType.numberWithOptions(
                                  decimal: true, signed: true),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                                    TextField(
                    controller: _textSizeController,
                    style:
                        const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Letter size (px on board)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                        decimal: true, signed: false),
                  ),
                  const SizedBox(height: 4),
                  // NEW: letter gap factor
                  TextField(
                    controller: _textGapController,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Letter gap (px)',
                      labelStyle: TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                        decimal: true, signed: false),
                  ),

                  const SizedBox(height: 4),
                  ElevatedButton(
                    onPressed: _writeTextFromUi,
                    child: const Text('Write text'),
                  ),

                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),

                  // --------------- GLOBAL SPEED ---------------
                  const Text(
                    'Global speed',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Multiplier: ${_globalSpeedMultiplier.toStringAsFixed(2)}x',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _globalSpeedMultiplier,
                    min: 0.25,
                    max: 3.0,
                    divisions: 11,
                    label: _globalSpeedMultiplier.toStringAsFixed(2),
                    onChanged: (v) {
                      setState(() {
                        _globalSpeedMultiplier = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),

                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),

                  // --------------- STROKE TIMING ---------------
                  const Text(
                    'Stroke timing (per stroke)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Min stroke time: ${_minStrokeTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _minStrokeTimeSec,
                    min: 0.05,
                    max: _maxStrokeTimeSec,
                    divisions: 50,
                    label: _minStrokeTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _minStrokeTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Max stroke time: ${_maxStrokeTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _maxStrokeTimeSec,
                    min: _minStrokeTimeSec,
                    max: 0.8,
                    divisions: 70,
                    label: _maxStrokeTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _maxStrokeTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Length factor: ${_lengthTimePerKPxSec.toStringAsFixed(3)} s per 1000px',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _lengthTimePerKPxSec,
                    min: 0.0,
                    max: 0.3,
                    divisions: 30,
                    label: _lengthTimePerKPxSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _lengthTimePerKPxSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Max curvature extra: ${_curvatureExtraMaxSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _curvatureExtraMaxSec,
                    min: 0.0,
                    max: 0.3,
                    divisions: 30,
                    label: _curvatureExtraMaxSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _curvatureExtraMaxSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),

                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),

                  // --------------- CURVATURE PROFILE ---------------
                  const Text(
                    'Curvature profile (within stroke)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Profile strength: ${_curvatureProfileFactor.toStringAsFixed(2)}x',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _curvatureProfileFactor,
                    min: 0.0,
                    max: 3.0,
                    divisions: 30,
                    label: _curvatureProfileFactor.toStringAsFixed(2),
                    onChanged: (v) {
                      setState(() {
                        _curvatureProfileFactor = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Angle scale: ${_curvatureAngleScale.toStringAsFixed(0)}°',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _curvatureAngleScale,
                    min: 20.0,
                    max: 160.0,
                    divisions: 14,
                    label: _curvatureAngleScale.toStringAsFixed(0),
                    onChanged: (v) {
                      setState(() {
                        _curvatureAngleScale = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),

                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),

                  // --------------- TRAVEL TIMING ---------------
                  const Text(
                    'Travel between strokes (non-text)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Base travel: ${_baseTravelTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _baseTravelTimeSec,
                    min: 0.0,
                    max: 0.6,
                    divisions: 60,
                    label: _baseTravelTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _baseTravelTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Distance factor: ${_travelTimePerKPxSec.toStringAsFixed(3)} s per 1000px',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _travelTimePerKPxSec,
                    min: 0.0,
                    max: 0.4,
                    divisions: 40,
                    label: _travelTimePerKPxSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _travelTimePerKPxSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Travel clamp min: ${_minTravelTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _minTravelTimeSec,
                    min: 0.0,
                    max: _maxTravelTimeSec,
                    divisions: 60,
                    label: _minTravelTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _minTravelTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Travel clamp max: ${_maxTravelTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _maxTravelTimeSec,
                    min: _minTravelTimeSec,
                    max: 0.8,
                    divisions: 80,
                    label: _maxTravelTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _maxTravelTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),

                  // --------------- ERASE UI ---------------
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Erase objects / text prompts',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  if (_drawnJsonNames.isEmpty)
                    const Text(
                      'No objects drawn yet.',
                      style:
                          TextStyle(color: Colors.white54, fontSize: 11),
                    )
                  else ...[
                    DropdownButton<String>(
                      dropdownColor: const Color(0xFF222222),
                      value: _selectedEraseName,
                      items: _drawnJsonNames
                          .map(
                            (name) => DropdownMenuItem(
                              value: name,
                              child: Text(
                                name,
                                style: const TextStyle(
                                    color: Colors.white, fontSize: 12),
                              ),
                            ),
                          )
                          .toList(),
                      onChanged: (v) {
                        setState(() {
                          _selectedEraseName = v;
                        });
                      },
                    ),
                    const SizedBox(height: 4),
                    ElevatedButton(
                      onPressed: _selectedEraseName == null
                          ? null
                          : () => _eraseObjectByName(_selectedEraseName!),
                      child: const Text('Erase selected'),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ------------------- GEOMETRY HELPERS -------------------

  Rect _computeRawBounds(
      List<StrokePolyline> polys, List<StrokeCubic> cubics) {
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
        for (final p in [seg.p0, seg.c1, seg.c2, seg.p1]) {
          if (p.dx < minX) minX = p.dx;
          if (p.dy < minY) minY = p.dy;
          if (p.dx > maxX) maxX = p.dx;
          if (p.dy > maxY) maxY = p.dy;
        }
      }
    }

    if (minX == double.infinity) {
      return const Rect.fromLTWH(0, 0, 1, 1);
    }
    final w = math.max(1e-3, maxX - minX);
    final h = math.max(1e-3, maxY - minY);
    return Rect.fromLTWH(minX, minY, w, h);
  }

  List<Offset> _downsamplePolyline(List<Offset> pts, int maxPoints) {
    final n = pts.length;
    if (n <= maxPoints || maxPoints <= 2) return pts;

    double totalLen = 0.0;
    final segLen = List<double>.filled(n, 0.0);

    for (int i = 1; i < n; i++) {
      final d = (pts[i] - pts[i - 1]).distance;
      segLen[i] = d;
      totalLen += d;
    }

    if (totalLen <= 1e-6) {
      return <Offset>[pts.first, pts.last];
    }

    final out = <Offset>[];
    out.add(pts.first);

    final step = totalLen / (maxPoints - 1);
    double accum = 0.0;
    double nextTarget = step;
    int i = 1;

    while (i < n - 1 && out.length < maxPoints - 1) {
      final d = segLen[i];
      if (d <= 0.0) {
        i++;
        continue;
      }

      if (accum + d >= nextTarget) {
        final t = (nextTarget - accum) / d;
        final p = Offset(
          pts[i - 1].dx + (pts[i].dx - pts[i - 1].dx) * t,
          pts[i - 1].dy + (pts[i].dy - pts[i - 1].dy) * t,
        );
        out.add(p);
        nextTarget += step;
      } else {
        accum += d;
        i++;
      }
    }

    out.add(pts.last);
    return out;
  }

  // ---------- BUILD DRAWABLE STROKES FOR ONE OBJECT (IMAGES) ----------

  List<DrawableStroke> _buildDrawableStrokesForObject({
    required String jsonName,
    required Offset origin,
    required double objectScale,
    required List<StrokePolyline> polylines,
    required List<StrokeCubic> cubics,
    required double srcWidth,
    required double srcHeight,
    required double targetResolution,
    required double basePenWidth,
  }) {
    final strokes = <DrawableStroke>[];

    final srcMax = math.max(srcWidth, srcHeight);
    final baseUpscale = srcMax > 0 ? targetResolution / srcMax : 1.0;
    final scale = objectScale <= 0 ? 1.0 : objectScale;
    final upscale = baseUpscale * scale;

    final diag = math.sqrt(
      math.pow(srcWidth * upscale, 2) + math.pow(srcHeight * upscale, 2),
    );
    final diagSafe = diag > 1e-3 ? diag : 1.0;

    final srcBounds = _computeRawBounds(polylines, cubics);
    final srcCenter = Offset(
      srcBounds.left + srcBounds.width / 2.0,
      srcBounds.top + srcBounds.height / 2.0,
    );
    final centerScaled = Offset(
      srcCenter.dx * upscale,
      srcCenter.dy * upscale,
    );

    for (final s in polylines) {
      final pts = s.points
          .map((p) {
            final scaled = Offset(p.dx * upscale, p.dy * upscale);
            return Offset(
              scaled.dx - centerScaled.dx + origin.dx,
              scaled.dy - centerScaled.dy + origin.dy,
            );
          })
          .toList(growable: false);

      if (pts.length < 2) continue;

      strokes.add(_makeDrawableFromPoints(
        jsonName: jsonName,
        objectOrigin: origin,
        objectScale: scale,
        pts: pts,
        basePenWidth: basePenWidth,
        diag: diagSafe,
      ));
    }

    for (final c in cubics) {
      final ptsRaw = _sampleCubicStroke(c, upscale: upscale);
      final pts = ptsRaw
          .map((p) => Offset(
                p.dx - centerScaled.dx + origin.dx,
                p.dy - centerScaled.dy + origin.dy,
              ))
          .toList(growable: false);

      if (pts.length < 2) continue;

      strokes.add(_makeDrawableFromPoints(
        jsonName: jsonName,
        objectOrigin: origin,
        objectScale: scale,
        pts: pts,
        basePenWidth: basePenWidth,
        diag: diagSafe,
      ));
    }

    return strokes;
  }

  DrawableStroke _makeDrawableFromPoints({
    required String jsonName,
    required Offset objectOrigin,
    required double objectScale,
    required List<Offset> pts,
    required double basePenWidth,
    required double diag,
  }) {
    final scale = objectScale <= 0 ? 1.0 : objectScale;

    final clampedScale = scale.clamp(0.1, 3.0);
    double scaleFactor;
    if (clampedScale <= 1.0) {
      scaleFactor = clampedScale;
    } else {
      scaleFactor = 1.0 + 0.4 * (clampedScale - 1.0);
    }

    int effectiveMax = (_maxDisplayPointsPerStroke * scaleFactor).round();
    if (effectiveMax < 8) effectiveMax = 8;
    if (effectiveMax > _maxDisplayPointsPerStroke) {
      effectiveMax = _maxDisplayPointsPerStroke;
    }

    final workPts = _downsamplePolyline(pts, effectiveMax);
    final n = workPts.length;

    final cumGeom = List<double>.filled(n, 0.0);
    final cumCost = List<double>.filled(n, 0.0);

    double length = 0.0;
    double cost = 0.0;

    double prevSharpNorm = 0.0;
    final double angleScale =
        _curvatureAngleScale.abs() < 1e-3 ? 1.0 : _curvatureAngleScale;

    for (int i = 1; i < n; i++) {
      final v = workPts[i] - workPts[i - 1];
      final segLen = v.distance;
      if (segLen < 1e-6) {
        cumGeom[i] = length;
        cumCost[i] = cost;
        continue;
      }

      length += segLen;

      double angDeg = 0.0;
      if (i > 1) {
        final vPrev = workPts[i - 1] - workPts[i - 2];
        final lenPrev = vPrev.distance;
        if (lenPrev >= 1e-6) {
          final dot =
              (vPrev.dx * v.dx + vPrev.dy * v.dy) / (lenPrev * segLen);
          final clamped = dot.clamp(-1.0, 1.0);
          angDeg = math.acos(clamped) * 180.0 / math.pi;
        }
      }

      double sharpNorm = (angDeg / angleScale).clamp(0.0, 1.5);

      final smoothedSharp = 0.7 * prevSharpNorm + 0.3 * sharpNorm;
      prevSharpNorm = smoothedSharp;

      final slowFactor = 1.0 + _curvatureProfileFactor * smoothedSharp;

      final segCost = segLen * slowFactor;

      cost += segCost;
      cumGeom[i] = length;
      cumCost[i] = cost;
    }

    if (length < 0.0) {
      length = 0.0;
    }

    double drawCostTotal;
    if (cost <= 0.0) {
      if (n > 1) {
        for (int i = 1; i < n; i++) {
          final t = i / (n - 1);
          cumGeom[i] = length * t;
          cumCost[i] = t;
        }
      }
      drawCostTotal = 1.0;
    } else {
      drawCostTotal = cost;
    }

    final centroid = _centroid(workPts);
    final curvature = _estimateCurvatureDeg(workPts);
    final bounds = _boundsOfPoints(workPts);

    double amp = 0.0;
    if (length > 0.02 * diag) {
      final lenNorm = (length / diag).clamp(0.0, 1.0);
      final curvNorm = (curvature / 70.0).clamp(0.0, 1.0);
      final baseAmp = basePenWidth * 0.9;
      amp = baseAmp *
          (0.5 + 0.8 * math.pow(lenNorm, 0.7)) *
          (0.6 + 0.4 * (1.0 - curvNorm));
      amp = amp.clamp(0.5, basePenWidth * 2.0);
    }

    final displayPts = amp > 0.0 ? _applyWobble(workPts, amp) : workPts;

    final lengthK = length / 1000.0;
    final curvNormGlobal = (curvature / 70.0).clamp(0.0, 1.0);

    final rawTime = _minStrokeTimeSec +
        lengthK * _lengthTimePerKPxSec +
        curvNormGlobal * _curvatureExtraMaxSec;

    final drawTimeSec =
        rawTime.clamp(_minStrokeTimeSec, _maxStrokeTimeSec).toDouble();

    return DrawableStroke(
      jsonName: jsonName,
      objectOrigin: objectOrigin,
      objectScale: objectScale,
      points: displayPts,
      originalPoints: workPts,
      lengthPx: length,
      centroid: centroid,
      bounds: bounds,
      curvatureMetricDeg: curvature,
      cumGeomLen: cumGeom,
      cumDrawCost: cumCost,
      drawCostTotal: drawCostTotal,
      drawTimeSec: drawTimeSec,
    );
  }

  Offset _centroid(List<Offset> pts) {
    if (pts.isEmpty) return Offset.zero;
    double sx = 0.0, sy = 0.0;
    for (final p in pts) {
      sx += p.dx;
      sy += p.dy;
    }
    final n = pts.length.toDouble();
    return Offset(sx / n, sy / n);
  }

  double _estimateCurvatureDeg(List<Offset> pts) {
    if (pts.length < 3) return 0.0;
    double sumAng = 0.0;
    int cnt = 0;
    for (int i = 1; i < pts.length - 1; i++) {
      final a = pts[i - 1];
      final b = pts[i];
      final c = pts[i + 1];
      final v1 = b - a;
      final v2 = c - b;
      final len1 = v1.distance;
      final len2 = v2.distance;
      if (len1 < 1e-3 || len2 < 1e-3) continue;
      final dot = (v1.dx * v2.dx + v1.dy * v2.dy) / (len1 * len2);
      final clamped = dot.clamp(-1.0, 1.0);
      final ang = math.acos(clamped) * 180.0 / math.pi;
      sumAng += ang.abs();
      cnt++;
    }
    if (cnt == 0) return 0.0;
    return sumAng / cnt;
  }

  List<Offset> _applyWobble(List<Offset> pts, double amp) {
    final n = pts.length;
    if (n < 3) return pts;

    final out = <Offset>[];
    for (int i = 0; i < n; i++) {
      final pPrev = pts[i == 0 ? 0 : i - 1];
      final pNext = pts[i == n - 1 ? n - 1 : i + 1];
      final dir = pNext - pPrev;
      final len = dir.distance;
      if (len < 1e-6) {
        out.add(pts[i]);
        continue;
      }
      final nx = -dir.dy / len;
      final ny = dir.dx / len;

      final t = i / (n - 1);
      final fade = math.sin(t * math.pi);

      final waveFast = math.sin(t * 5.0 * math.pi);
      final waveSlow = math.sin(t * 2.0 * math.pi);
      final combined = 0.6 * waveFast + 0.4 * waveSlow;

      final w = combined * amp * fade;

      final dx = nx * w;
      final dy = ny * w;
      out.add(Offset(pts[i].dx + dx, pts[i].dy + dy));
    }
    return out;
  }

  Rect _boundsOfPoints(List<Offset> pts) {
    double minX = double.infinity, minY = double.infinity;
    double maxX = -double.infinity, maxY = -double.infinity;
    for (final p in pts) {
      if (p.dx < minX) minX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy > maxY) maxY = p.dy;
    }
    if (minX == double.infinity) {
      return const Rect.fromLTWH(0, 0, 1, 1);
    }
    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  List<Offset> _sampleCubicStroke(StrokeCubic s, {required double upscale}) {
    const int stepsPerSegment = 18;
    final pts = <Offset>[];
    bool first = true;

    for (final seg in s.segments) {
      for (int i = 0; i <= stepsPerSegment; i++) {
        final t = i / stepsPerSegment;
        final p = _evalCubic(seg, t);
        final q = Offset(p.dx * upscale, p.dy * upscale);
        if (!first) {
          if ((q - pts.last).distance < 0.05) continue;
        }
        pts.add(q);
        first = false;
      }
    }
    return pts;
  }

  Offset _evalCubic(CubicSegment seg, double t) {
    final mt = 1.0 - t;
    final mt2 = mt * mt;
    final t2 = t * t;
    final x = mt2 * mt * seg.p0.dx +
        3 * mt2 * t * seg.c1.dx +
        3 * mt * t2 * seg.c2.dx +
        t2 * t * seg.p1.dx;
    final y = mt2 * mt * seg.p0.dy +
        3 * mt2 * t * seg.c1.dy +
        3 * mt * t2 * seg.c2.dy +
        t2 * t * seg.p1.dy;
    return Offset(x, y);
  }
}

/* ---------- Raw data types ---------- */

class StrokePolyline {
  final List<Offset> points;
  const StrokePolyline(this.points);
}

class CubicSegment {
  final Offset p0;
  final Offset c1;
  final Offset c2;
  final Offset p1;
  const CubicSegment({
    required this.p0,
    required this.c1,
    required this.c2,
    required this.p1,
  });
}

class StrokeCubic {
  final List<CubicSegment> segments;
  const StrokeCubic(this.segments);
}

class GlyphData {
  final List<StrokeCubic> cubics;
  final Rect bounds;
  const GlyphData({
    required this.cubics,
    required this.bounds,
  });
}

/* ---------- Drawable stroke ---------- */

class DrawableStroke {
  // origin + identity for erase / grouping
  final String jsonName;
  final Offset objectOrigin;
  final double objectScale;

  final List<Offset> points; // upscaled + wobble + placed
  final List<Offset> originalPoints; // pre-wobble (downsampled, with placement)
  final double lengthPx;
  final Offset centroid;
  final Rect bounds;
  final double curvatureMetricDeg;

  final List<double> cumGeomLen;
  final List<double> cumDrawCost;
  final double drawCostTotal;

  double drawTimeSec; // how long we draw this stroke
  double travelTimeBeforeSec; // pause/travel before this stroke starts
  double timeWeight; // travel + draw; used by global animator

  int groupId = -1;
  int groupSize = 1;
  double importanceScore = 0.0;

  DrawableStroke({
    required this.jsonName,
    required this.objectOrigin,
    required this.objectScale,
    required this.points,
    required this.originalPoints,
    required this.lengthPx,
    required this.centroid,
    required this.bounds,
    required this.curvatureMetricDeg,
    required this.cumGeomLen,
    required this.cumDrawCost,
    required this.drawCostTotal,
    required this.drawTimeSec,
    this.travelTimeBeforeSec = 0.0,
    this.timeWeight = 0.0,
  });
}

/* ---------- Painter ---------- */

class WhiteboardPainter extends CustomPainter {
  final List<DrawableStroke> staticStrokes;
  final List<DrawableStroke> animStrokes;
  final double animationT;
  final double basePenWidth;

  final bool stepMode;
  final int stepStrokeCount;

  // Virtual board size – used to keep 0,0 stable.
  final double boardWidth;
  final double boardHeight;

  const WhiteboardPainter({
    required this.staticStrokes,
    required this.animStrokes,
    required this.animationT,
    required this.basePenWidth,
    required this.stepMode,
    required this.stepStrokeCount,
    required this.boardWidth,
    required this.boardHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (staticStrokes.isEmpty && animStrokes.isEmpty) return;

    final allStrokes = <DrawableStroke>[
      ...staticStrokes,
      ...animStrokes,
    ];

    final bounds = _computeBounds(allStrokes);
    final scale = _computeUniformScale(bounds, size, padding: 80);
    final tx =
        (size.width - bounds.width * scale) / 2 - bounds.left * scale;
    final ty =
        (size.height - bounds.height * scale) / 2 - bounds.top * scale;

    canvas.save();
    canvas.translate(tx, ty);
    canvas.scale(scale);

    if (stepMode) {
      final count = stepStrokeCount.clamp(0, allStrokes.length);
      for (int i = 0; i < count; i++) {
        _drawStroke(canvas, allStrokes[i], 1.0, scale);
      }
    } else {
      for (final s in staticStrokes) {
        _drawStroke(canvas, s, 1.0, scale);
      }

      if (animStrokes.isNotEmpty) {
        final totalWeight =
            animStrokes.fold<double>(0.0, (s, d) => s + d.timeWeight);
        final clampedT = animationT.clamp(0.0, 1.0);
        final target = totalWeight > 0 ? totalWeight * clampedT : 0.0;

        double acc = 0.0;
        for (final stroke in animStrokes) {
          final travel = stroke.travelTimeBeforeSec;
          final draw = stroke.drawTimeSec;
          if (draw <= 0.0 && travel <= 0.0) {
            continue;
          }

          final strokeStart = acc;
          final travelEnd = strokeStart + travel;
          final strokeEnd = travelEnd + draw;
          acc = strokeEnd;

          if (target >= strokeEnd) {
            _drawStroke(canvas, stroke, 1.0, scale);
            continue;
          }

          if (target <= strokeStart) {
            break;
          }

          if (target < travelEnd) {
            break;
          }

          final local = (target - travelEnd) / draw;
          final phase = local.clamp(0.0, 1.0);
          if (phase > 0.0) {
            _drawStroke(canvas, stroke, phase, scale);
          }
          break;
        }
      }
    }

    canvas.restore();
  }

  void _drawStroke(
      Canvas canvas, DrawableStroke stroke, double phase, double viewScale) {
    final pts = stroke.points;
    if (pts.length < 2) return;

    const double drawFrac = 0.8;
    final local = phase.clamp(0.0, 1.0);
    if (local <= 0.0) return;

    double drawPhase;
    if (local >= drawFrac) {
      drawPhase = 1.0;
    } else {
      drawPhase = local / drawFrac;
    }

    if (drawPhase <= 0.0) return;

    final n = pts.length;
    final totalCost = stroke.drawCostTotal;

    int idxMax;
    if (drawPhase >= 1.0 || totalCost <= 0.0) {
      idxMax = n - 1;
    } else {
      final targetCost = drawPhase * totalCost;
      idxMax = _findIndexForCost(stroke.cumDrawCost, targetCost);
      if (idxMax < 1) idxMax = 1;
      if (idxMax >= n) idxMax = n - 1;
    }

    if (idxMax < 1) return;

    final path = Path()..moveTo(pts[0].dx, pts[0].dy);
    for (int i = 1; i <= idxMax; i++) {
      path.lineTo(pts[i].dx, pts[i].dy);
    }

    final penW = (basePenWidth / viewScale).clamp(0.5, 10.0);

    final paintLine = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = penW;

    canvas.drawPath(path, paintLine);
  }

  int _findIndexForCost(List<double> cumCost, double target) {
    final last = cumCost.length - 1;
    if (last <= 0) return 0;
    if (target >= cumCost[last]) return last;

    int lo = 1;
    int hi = last;
    int ans = 1;
    while (lo <= hi) {
      final mid = (lo + hi) >> 1;
      if (cumCost[mid] <= target) {
        ans = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans;
  }

  Rect _computeBounds(List<DrawableStroke> strokes) {
    final double halfW = boardWidth / 2.0;
    final double halfH = boardHeight / 2.0;
    return Rect.fromLTWH(-halfW, -halfH, boardWidth, boardHeight);
  }

  double _computeUniformScale(Rect bounds, Size size,
      {double padding = 10}) {
    final sx = (size.width - 2 * padding) / bounds.width;
    final sy = (size.height - 2 * padding) / bounds.height;
    final v = math.min(sx, sy);
    final fit = (v.isFinite && v > 0) ? v : 1.0;
    const shrinkFactor = 0.45;
    return fit * shrinkFactor;
  }

  @override
  bool shouldRepaint(covariant WhiteboardPainter old) =>
      old.staticStrokes != staticStrokes ||
      old.animStrokes != animStrokes ||
      old.animationT != animationT ||
      old.basePenWidth != basePenWidth ||
      old.stepMode != stepMode ||
      old.stepStrokeCount != stepStrokeCount ||
      old.boardWidth != boardWidth ||
      old.boardHeight != boardHeight;
}
