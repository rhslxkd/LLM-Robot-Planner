with open("run_random_batch_v2.py") as f:
    content = f.read()

old = '''            layout_rng = random.Random(seed * 7919 + 13)
            use_boxes = layout_rng.random() < args.box_prob
            if use_boxes:
                xml, meta = generate_boxes(seed, scene)
            else:
                xml, meta = generate_slalom(seed, scene)'''

new = '''            layout_rng = random.Random(seed * 7919 + 13)
            use_boxes = layout_rng.random() < args.box_prob
            if use_boxes:
                room_half_y = layout_rng.uniform(2.0, 4.0)
                goal_x_hi = 4.4
            else:
                room_half_y = layout_rng.uniform(1.4, 2.4)
                goal_x_hi = 4.6
            goal_y_margin = room_half_y * 0.3
            goal_y_bound = room_half_y - goal_y_margin
            goal_x_range = (1.0, goal_x_hi)
            max_dist = (goal_x_hi ** 2 + goal_y_bound ** 2) ** 0.5
            min_dist_from_start = layout_rng.uniform(0.3, 0.85) * max_dist
            diversity_kwargs = dict(
                goal_x_range=goal_x_range,
                room_half_y=room_half_y,
                goal_y_margin=goal_y_margin,
                min_dist_from_start=min_dist_from_start,
            )
            if use_boxes:
                xml, meta = generate_boxes(seed, scene, **diversity_kwargs)
            else:
                xml, meta = generate_slalom(seed, scene, **diversity_kwargs)'''

count = content.count(old)
assert count == 1, f"old occurrence = {count}"
content = content.replace(old, new)

with open("run_random_batch_v2.py", "w") as f:
    f.write(content)

print("패치 완료")
